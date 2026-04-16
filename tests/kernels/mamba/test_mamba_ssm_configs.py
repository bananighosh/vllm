# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the JSON-based config loader added to selective_state_update.

Tests cover:
  - Config filename generation
  - Loading a bundled config file
  - VLLM_TUNED_CONFIG_FOLDER env-var override
  - Fallback to heuristic when no config file exists
"""

import json
import os

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _get_ssm_launch_config,
    get_ssm_config_file_name,
    get_ssm_configs,
)


# ---------------------------------------------------------------------------
# Config filename generation
# ---------------------------------------------------------------------------

def test_config_file_name_format():
    name = get_ssm_config_file_name(128)
    # Must start with dstate= and contain device_name=
    assert name.startswith("dstate=128,device_name=")
    assert name.endswith(".json")
    # Spaces must be replaced with underscores (GPU names have spaces)
    assert " " not in name


# ---------------------------------------------------------------------------
# VLLM_TUNED_CONFIG_FOLDER override
# ---------------------------------------------------------------------------

def test_env_override_loads_custom_config(monkeypatch, tmp_path):
    """VLLM_TUNED_CONFIG_FOLDER should take precedence over the bundled dir."""
    file_name = get_ssm_config_file_name(16)
    config_path = os.path.join(tmp_path, file_name)
    payload = {"1": {"BLOCK_SIZE_M": 4, "num_warps": 1}}
    with open(config_path, "w") as f:
        json.dump(payload, f)

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    get_ssm_configs.cache_clear()

    cfg = get_ssm_configs(16)
    assert cfg is not None
    assert cfg[1] == {"BLOCK_SIZE_M": 4, "num_warps": 1}

    get_ssm_configs.cache_clear()


# ---------------------------------------------------------------------------
# Fallback to heuristic when no config file exists
# ---------------------------------------------------------------------------

def test_fallback_when_no_config(monkeypatch, tmp_path):
    """_get_ssm_launch_config must fall back to the hard-coded heuristic
    when no JSON file is found for the current device."""
    get_ssm_configs.cache_clear()

    # Point config folder at an empty directory so no file is found.
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    # Also shadow the bundled configs dir so it cannot match either.
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm._CONFIGS_DIR",
        str(tmp_path),
    )
    get_ssm_configs.cache_clear()

    # dstate=64 heuristic: BLOCK_SIZE_M=8, num_warps=4
    block_m, warps = _get_ssm_launch_config(dstate=64, batch=1,
                                             is_blackwell=False)
    assert block_m == 8
    assert warps == 4

    # dstate=16 heuristic: BLOCK_SIZE_M=32, num_warps=4
    block_m, warps = _get_ssm_launch_config(dstate=16, batch=1,
                                             is_blackwell=False)
    assert block_m == 32
    assert warps == 4

    get_ssm_configs.cache_clear()


# ---------------------------------------------------------------------------
# Nearest-batch interpolation
# ---------------------------------------------------------------------------

def test_nearest_batch_interpolation(monkeypatch, tmp_path):
    """When the exact batch size is not in the config, the closest key
    should be selected."""
    get_ssm_configs.cache_clear()

    file_name = get_ssm_config_file_name(32)
    config_path = os.path.join(tmp_path, file_name)
    # Only provide configs for batch 1 and 64.
    payload = {
        "1":  {"BLOCK_SIZE_M": 8,  "num_warps": 1},
        "64": {"BLOCK_SIZE_M": 32, "num_warps": 4},
    }
    with open(config_path, "w") as f:
        json.dump(payload, f)

    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    get_ssm_configs.cache_clear()

    # batch=5 is closer to 1 than to 64 — expects M=8, w=1
    block_m, warps = _get_ssm_launch_config(dstate=32, batch=5,
                                             is_blackwell=False)
    assert block_m == 8 and warps == 1

    # batch=40 is closer to 64 — expects M=32, w=4
    block_m, warps = _get_ssm_launch_config(dstate=32, batch=40,
                                             is_blackwell=False)
    assert block_m == 32 and warps == 4

    get_ssm_configs.cache_clear()
