# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark script for Cascade adaptive-k speculative decoding.

Compares static-k baseline against Cascade across a sweep of:
  - k_max values (number of speculative tokens)
  - cascade_cost_factor values
  - input/output length combinations

Results are saved to JSON (machine-readable) and a Markdown table
(human-readable for PR reviewers).

Usage
-----
# Quick smoke test (small model, no GPU required except for the model itself):
python benchmarks/benchmark_cascade_spec_decode.py \\
    --model <target-model> \\
    --speculative-model <draft-model> \\
    --method deepseek_mtp \\
    --output-dir ./cascade_results

# Full MoE sweep (e.g. DeepSeek-V3 with MTP on B200):
python benchmarks/benchmark_cascade_spec_decode.py \\
    --model deepseek-ai/DeepSeek-V3 \\
    --speculative-model deepseek-ai/DeepSeek-V3 \\
    --method deepseek_mtp \\
    --k-max-values 1 2 4 \\
    --cost-factors 0.1 0.3 0.5 0.8 \\
    --input-len 512 \\
    --output-len 256 \\
    --num-prompts 200 \\
    --output-dir ./cascade_results_b200 \\
    --tp 8
"""

import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Vector

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Parameters for a single benchmark run."""

    mode: str  # "static" or "cascade"
    k_max: int
    cost_factor: float | None = None  # None for static mode
    cascade_steps_per_k: int = 20
    cascade_re_test_interval: int = 0


@dataclass
class RunResult:
    """Metrics collected from one benchmark run."""

    config: RunConfig
    throughput_toks_per_s: float = 0.0
    latency_s: float = 0.0
    num_prompts: int = 0
    num_output_tokens: int = 0
    mean_acceptance_length: float = 0.0
    acceptance_rate_pct: float = 0.0
    per_pos_acceptance: list[float] = field(default_factory=list)
    num_drafts: int = 0
    num_accepted_tokens: int = 0
    num_draft_tokens: int = 0
    gpu_name: str = ""
    vllm_version: str = ""
    python_version: str = ""
    torch_version: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_gpu_name() -> str:
    try:
        return current_platform.get_device_name(0)
    except Exception:
        return "unknown"


def _get_vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return "unknown"


def _get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _synthetic_prompts(num_prompts: int, input_len: int) -> list[list[int]]:
    """Generate random token-id prompts (no tokenizer needed)."""
    import numpy as np

    rng = np.random.default_rng(42)
    return [rng.integers(1, 32000, size=input_len).tolist() for _ in range(num_prompts)]


def _extract_spec_metrics(metrics, k_max: int) -> dict:
    """Pull spec-decode counters out of the Prometheus metric snapshot."""
    num_drafts = 0
    num_accepted = 0
    num_drafted = 0
    per_pos: list[float] = []

    for m in metrics:
        if m.name == "vllm:spec_decode_num_drafts" and isinstance(m, Counter):
            num_drafts += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(
            m, Counter
        ):
            num_accepted += m.value
        elif m.name == "vllm:spec_decode_num_draft_tokens" and isinstance(m, Counter):
            num_drafted += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and isinstance(
            m, Vector
        ):
            per_pos = list(m.values[:k_max])

    mean_al = 1 + (num_accepted / num_drafts) if num_drafts > 0 else 1.0
    acceptance_rate = (num_accepted / num_drafted * 100) if num_drafted > 0 else 0.0
    per_pos_norm = [v / num_drafts if num_drafts > 0 else 0.0 for v in per_pos]

    return {
        "num_drafts": num_drafts,
        "num_accepted_tokens": num_accepted,
        "num_draft_tokens": num_drafted,
        "mean_acceptance_length": mean_al,
        "acceptance_rate_pct": acceptance_rate,
        "per_pos_acceptance": per_pos_norm,
    }


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def run_single(
    cfg: RunConfig,
    model: str,
    speculative_model: str,
    method: str,
    prompts: list[list[int]],
    output_len: int,
    tp: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    temperature: float,
) -> RunResult:
    """Run one configuration and return collected metrics."""
    spec_cfg: dict = {
        "method": method,
        "model": speculative_model,
        "num_speculative_tokens": cfg.k_max,
    }
    if cfg.mode == "cascade":
        assert cfg.cost_factor is not None
        spec_cfg["enable_cascade"] = True
        spec_cfg["cascade_cost_factor"] = cfg.cost_factor
        spec_cfg["cascade_steps_per_k"] = cfg.cascade_steps_per_k
        spec_cfg["cascade_re_test_interval"] = cfg.cascade_re_test_interval

    llm_kwargs: dict = dict(
        model=model,
        speculative_config=spec_cfg,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=False,
        max_model_len=max_model_len,
    )

    print(
        f"\n{'=' * 70}\n"
        f"  mode={cfg.mode}  k_max={cfg.k_max}  "
        f"cost_factor={cfg.cost_factor}\n"
        f"{'=' * 70}"
    )

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=output_len,
    )
    token_prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts]

    # Warm-up (small batch)
    llm.generate(token_prompts[:4], sampling_params)

    t0 = time.perf_counter()
    outputs = llm.generate(token_prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_output_tokens / elapsed

    metrics = llm.get_metrics()
    spec_stats = _extract_spec_metrics(metrics, cfg.k_max)

    result = RunResult(
        config=cfg,
        throughput_toks_per_s=throughput,
        latency_s=elapsed,
        num_prompts=len(prompts),
        num_output_tokens=total_output_tokens,
        gpu_name=_get_gpu_name(),
        vllm_version=_get_vllm_version(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        **spec_stats,
    )

    # Explicitly free GPU memory before next run.
    del llm
    torch.accelerator.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


@dataclass
class CostScalePoint:
    """Single data point from the verification-cost-scaling micro-benchmark."""

    k: int
    mean_step_ms: float
    std_step_ms: float
    relative_cost: float  # T(k) / T(0)


def profile_cost_scaling(
    model: str,
    speculative_model: str,
    method: str,
    k_max: int,
    num_steps: int,
    input_len: int,
    tp: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
) -> list[CostScalePoint]:
    """Measure per-step verification latency vs k to empirically fit cost_factor.

    Runs the engine with a single long-running request at each k value and
    records per-scheduler-step wall time.  The ratio T(k)/T(0) shows whether
    verification cost is constant (dense) or linear in k (MoE).

    Returns a list of CostScalePoint, one per k in range(0, k_max+1).
    """
    import statistics

    results: list[CostScalePoint] = []
    baseline_ms: float | None = None

    for k in range(0, k_max + 1):
        spec_cfg: dict = {
            "method": method,
            "model": speculative_model,
            "num_speculative_tokens": k if k > 0 else 1,
        }
        llm = LLM(
            model=model,
            speculative_config=spec_cfg if k > 0 else None,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            disable_log_stats=True,
        )

        import numpy as np

        rng = np.random.default_rng(0)
        prompt = rng.integers(1, 32000, size=input_len).tolist()
        sampling_params = SamplingParams(temperature=0.0, max_tokens=num_steps * 2)

        # Warm-up
        llm.generate([TokensPrompt(prompt_token_ids=prompt)], sampling_params)

        # Timed run: generate exactly num_steps tokens, record wall time
        step_times_ms: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            llm.generate(
                [TokensPrompt(prompt_token_ids=prompt)],
                SamplingParams(temperature=0.0, max_tokens=num_steps),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Per-step estimate: divide total time by num_steps * (k+1 effective)
            step_times_ms.append(elapsed_ms / num_steps)

        mean_ms = statistics.mean(step_times_ms)
        std_ms = statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0

        if k == 0:
            baseline_ms = mean_ms
        rel = mean_ms / baseline_ms if baseline_ms else 1.0

        results.append(
            CostScalePoint(
                k=k,
                mean_step_ms=round(mean_ms, 3),
                std_step_ms=round(std_ms, 3),
                relative_cost=round(rel, 4),
            )
        )

        print(f"  k={k}: {mean_ms:.2f} ms/step  (±{std_ms:.2f})  relative={rel:.3f}x")

        del llm
        torch.accelerator.empty_cache()

    return results


def save_cost_scaling(
    points: list[CostScalePoint],
    path_json: Path,
    path_md: Path,
    gpu_name: str,
    git_hash: str,
) -> None:
    """Save cost-scaling results as JSON + Markdown for reviewers."""
    # JSON
    path_json.write_text(
        json.dumps(
            [
                {
                    "k": p.k,
                    "mean_step_ms": p.mean_step_ms,
                    "std_step_ms": p.std_step_ms,
                    "relative_cost": p.relative_cost,
                }
                for p in points
            ],
            indent=2,
        )
    )
    print(f"Cost-scaling JSON → {path_json}")

    # Fit linear model: relative_cost ≈ 1 + k * cost_factor
    if len(points) >= 2:
        ks = [p.k for p in points if p.k > 0]
        rels = [p.relative_cost for p in points if p.k > 0]
        if ks:
            cost_factor_estimate = sum((r - 1) / k for k, r in zip(ks, rels)) / len(ks)
        else:
            cost_factor_estimate = 0.0
    else:
        cost_factor_estimate = 0.0

    lines = [
        "# Cascade: Verification-Cost Scaling Profile",
        "",
        f"**GPU:** {gpu_name}  ",
        f"**Commit:** `{git_hash}`  ",
        f"**Empirical cost_factor:** {cost_factor_estimate:.3f}  ",
        "(cost_factor = mean of (T(k)/T(0) − 1) / k across all k > 0)",
        "",
        "| k (draft tokens) | Mean step (ms) | ±std | Relative cost T(k)/T(0) |",
        "|-----------------|---------------|------|------------------------|",
    ]
    for p in points:
        lines.append(
            f"| {p.k} | {p.mean_step_ms:.2f} | "
            f"±{p.std_step_ms:.2f} | {p.relative_cost:.4f} |"
        )
    lines += [
        "",
        "> **Interpretation:** For a dense model, relative cost stays near 1.0 "
        "regardless of k.",
        "> For a MoE model, each additional draft token activates a different expert "
        "subset, so cost grows",
        "> roughly linearly. Use the empirical `cost_factor` above as the "
        "`--cascade-cost-factor` argument.",
    ]
    path_md.write_text("\n".join(lines) + "\n")
    print(f"Cost-scaling Markdown → {path_md}")


def save_json(results: list[RunResult], path: Path) -> None:
    rows = []
    for r in results:
        d = {
            "mode": r.config.mode,
            "k_max": r.config.k_max,
            "cost_factor": r.config.cost_factor,
            "cascade_steps_per_k": r.config.cascade_steps_per_k,
            "throughput_toks_per_s": r.throughput_toks_per_s,
            "latency_s": r.latency_s,
            "num_prompts": r.num_prompts,
            "num_output_tokens": r.num_output_tokens,
            "mean_acceptance_length": r.mean_acceptance_length,
            "acceptance_rate_pct": r.acceptance_rate_pct,
            "per_pos_acceptance": r.per_pos_acceptance,
            "num_drafts": r.num_drafts,
            "num_accepted_tokens": r.num_accepted_tokens,
            "num_draft_tokens": r.num_draft_tokens,
            "gpu_name": r.gpu_name,
            "vllm_version": r.vllm_version,
            "python_version": r.python_version,
            "torch_version": r.torch_version,
            "timestamp": r.timestamp,
        }
        rows.append(d)
    path.write_text(json.dumps(rows, indent=2))
    print(f"JSON results → {path}")


def save_markdown(results: list[RunResult], path: Path, git_hash: str) -> None:
    """Write a Markdown table suitable for pasting into a GitHub PR."""
    if not results:
        return

    gpu = results[0].gpu_name
    vllm_ver = results[0].vllm_version
    ts = results[0].timestamp

    lines = [
        "# Cascade Spec-Decode Benchmark Results",
        "",
        f"**GPU:** {gpu}  ",
        f"**vLLM:** {vllm_ver} (commit `{git_hash}`)  ",
        f"**Date:** {ts}  ",
        "",
        "## Throughput comparison",
        "",
        "| Mode | k_max | cost_factor | Throughput (tok/s) | Mean Acc. Len | "
        "Acc. Rate % | vs static Δ% |",
        "|------|-------|-------------|-------------------|--------------|"
        "------------|------------|",
    ]

    # Build a lookup: static results keyed by k_max
    static_map: dict[int, RunResult] = {}
    for r in results:
        if r.config.mode == "static":
            static_map[r.config.k_max] = r

    for r in results:
        mode = r.config.mode
        k = r.config.k_max
        cf = f"{r.config.cost_factor:.2f}" if r.config.cost_factor is not None else "—"
        tput = f"{r.throughput_toks_per_s:.1f}"
        mal = f"{r.mean_acceptance_length:.3f}"
        acc = f"{r.acceptance_rate_pct:.1f}"
        if mode == "cascade" and k in static_map:
            delta = (
                (r.throughput_toks_per_s - static_map[k].throughput_toks_per_s)
                / static_map[k].throughput_toks_per_s
                * 100
            )
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "baseline"
        lines.append(f"| {mode} | {k} | {cf} | {tput} | {mal} | {acc} | {delta_str} |")

    lines += [
        "",
        "## Per-position acceptance rates",
        "",
        "| Mode | k_max | cost_factor | "
        + " | ".join(
            f"pos {i}" for i in range(max(len(r.per_pos_acceptance) for r in results))
        )
        + " |",
        "|------|-------|-------------|"
        + "------|" * max(len(r.per_pos_acceptance) for r in results),
    ]
    for r in results:
        cf = f"{r.config.cost_factor:.2f}" if r.config.cost_factor is not None else "—"
        pos_cells = " | ".join(f"{v:.3f}" for v in r.per_pos_acceptance)
        lines.append(f"| {r.config.mode} | {r.config.k_max} | {cf} | {pos_cells} |")

    path.write_text("\n".join(lines) + "\n")
    print(f"Markdown report → {path}")


def save_csv(results: list[RunResult], path: Path) -> None:
    """Write a flat CSV for spreadsheet analysis."""
    import csv

    fieldnames = [
        "mode",
        "k_max",
        "cost_factor",
        "cascade_steps_per_k",
        "throughput_toks_per_s",
        "latency_s",
        "num_prompts",
        "num_output_tokens",
        "mean_acceptance_length",
        "acceptance_rate_pct",
        "num_drafts",
        "num_accepted_tokens",
        "num_draft_tokens",
        "gpu_name",
        "vllm_version",
        "timestamp",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "mode": r.config.mode,
                "k_max": r.config.k_max,
                "cost_factor": r.config.cost_factor,
                "cascade_steps_per_k": r.config.cascade_steps_per_k,
                "throughput_toks_per_s": r.throughput_toks_per_s,
                "latency_s": r.latency_s,
                "num_prompts": r.num_prompts,
                "num_output_tokens": r.num_output_tokens,
                "mean_acceptance_length": r.mean_acceptance_length,
                "acceptance_rate_pct": r.acceptance_rate_pct,
                "num_drafts": r.num_drafts,
                "num_accepted_tokens": r.num_accepted_tokens,
                "num_draft_tokens": r.num_draft_tokens,
                "gpu_name": r.gpu_name,
                "vllm_version": r.vllm_version,
                "timestamp": r.timestamp,
            }
            writer.writerow(row)
    print(f"CSV results → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark Cascade adaptive-k speculative decoding vs static-k"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Target model name or path (e.g. deepseek-ai/DeepSeek-V3)",
    )
    parser.add_argument(
        "--speculative-model",
        default=None,
        help="Draft model name/path. Defaults to --model (for MTP methods).",
    )
    parser.add_argument(
        "--method",
        default="deepseek_mtp",
        help="Speculative decoding method (default: deepseek_mtp)",
    )
    parser.add_argument(
        "--k-max-values",
        nargs="+",
        type=int,
        default=[4],
        metavar="K",
        help="List of k_max (num_speculative_tokens) values to sweep",
    )
    parser.add_argument(
        "--cost-factors",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5],
        metavar="CF",
        help="Cascade cost_factor values to sweep (0=dense, 1=pure-MoE)",
    )
    parser.add_argument(
        "--cascade-steps-per-k",
        type=int,
        default=20,
        help="Testing-phase steps per k (cascade_steps_per_k)",
    )
    parser.add_argument(
        "--cascade-re-test-interval",
        type=int,
        default=0,
        help="Production steps between re-tests (0=never)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=512,
        help="Synthetic prompt length in tokens",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=256,
        help="Max generated tokens per prompt",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts in the benchmark batch",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel size",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override max model length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--output-dir",
        default="./cascade_benchmark_results",
        help="Directory to save JSON / Markdown / CSV output",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix added to output file names",
    )
    parser.add_argument(
        "--profile-cost-scaling",
        action="store_true",
        help=(
            "Run a micro-benchmark that measures per-step latency at each k "
            "to empirically determine the verification cost_factor. "
            "Saves cost_scaling_<gpu>.json and cost_scaling_<gpu>.md."
        ),
    )
    parser.add_argument(
        "--cost-scaling-steps",
        type=int,
        default=50,
        help="Number of decoding steps per k during cost-scaling profiling",
    )
    args = parser.parse_args()

    spec_model = args.speculative_model or args.model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_hash = _get_git_hash()
    gpu_slug = _get_gpu_name().replace(" ", "_").replace("/", "-")
    tag = f"_{args.tag}" if args.tag else ""

    # ------------------------------------------------------------------
    # Optional: cost-scaling profiling (run first to inform cost_factor)
    # ------------------------------------------------------------------
    if args.profile_cost_scaling:
        k_max_for_profile = max(args.k_max_values)
        print(
            f"\n{'=' * 70}\n"
            f"  Cost-scaling profiling: k=0..{k_max_for_profile} "
            f"({args.cost_scaling_steps} steps each)\n"
            f"{'=' * 70}"
        )
        cost_points = profile_cost_scaling(
            model=args.model,
            speculative_model=spec_model,
            method=args.method,
            k_max=k_max_for_profile,
            num_steps=args.cost_scaling_steps,
            input_len=args.input_len,
            tp=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        stem_cs = f"cost_scaling_{gpu_slug}{tag}"
        save_cost_scaling(
            cost_points,
            path_json=out_dir / f"{stem_cs}.json",
            path_md=out_dir / f"{stem_cs}.md",
            gpu_name=_get_gpu_name(),
            git_hash=git_hash,
        )

    prompts = _synthetic_prompts(args.num_prompts, args.input_len)
    results: list[RunResult] = []

    # ------------------------------------------------------------------
    # Build the run plan: one static baseline per k, then cascade sweeps
    # ------------------------------------------------------------------
    run_plan: list[RunConfig] = []
    for k in args.k_max_values:
        run_plan.append(RunConfig(mode="static", k_max=k))
        for cf in args.cost_factors:
            run_plan.append(
                RunConfig(
                    mode="cascade",
                    k_max=k,
                    cost_factor=cf,
                    cascade_steps_per_k=args.cascade_steps_per_k,
                    cascade_re_test_interval=args.cascade_re_test_interval,
                )
            )

    print(
        f"\nPlan: {len(run_plan)} runs across {len(args.k_max_values)} k values "
        f"and {len(args.cost_factors)} cost factors."
    )

    stem = f"cascade_{gpu_slug}{tag}"
    for i, cfg in enumerate(run_plan):
        print(f"\n[{i + 1}/{len(run_plan)}] {cfg}")
        try:
            result = run_single(
                cfg=cfg,
                model=args.model,
                speculative_model=spec_model,
                method=args.method,
                prompts=prompts,
                output_len=args.output_len,
                tp=args.tp,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                temperature=args.temperature,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Save incrementally so partial results survive crashes.
        save_json(results, out_dir / f"{stem}.json")
        save_csv(results, out_dir / f"{stem}.csv")
        save_markdown(results, out_dir / f"{stem}.md", git_hash)

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
