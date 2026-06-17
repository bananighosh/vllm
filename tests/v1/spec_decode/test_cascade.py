# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Cascade adaptive-k speculative decoding."""

import torch

from vllm.v1.spec_decode.cascade import CascadeTracker, RequestCascadeState

# ---------------------------------------------------------------------------
# Helpers shared by scheduler integration tests
# ---------------------------------------------------------------------------


def _make_cascade_scheduler(
    num_spec_tokens: int = 4,
    cost_factor: float = 0.5,
    t_test: int = 5,
    re_test_interval: int = 0,
    num_blocks: int = 10000,
    block_size: int = 16,
):
    """Create a minimal Scheduler with Cascade enabled (ngram proposer)."""
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        enable_chunked_prefill=True,
        watermark=0.0,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    spec_config = SpeculativeConfig(
        model="ngram",
        num_speculative_tokens=num_spec_tokens,
        enable_cascade=True,
        cascade_cost_factor=cost_factor,
        cascade_steps_per_k=t_test,
        cascade_re_test_interval=re_test_interval,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=spec_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


# ---------------------------------------------------------------------------
# RequestCascadeState tests
# ---------------------------------------------------------------------------


class TestRequestCascadeState:
    def _make_state(self, k_max=4, t_test=5, cost_factor=0.5, re_test=0):
        return RequestCascadeState(
            k_max=k_max,
            t_test=t_test,
            cost_factor=cost_factor,
            re_test_interval=re_test,
        )

    def test_initial_state(self):
        s = self._make_state(k_max=4)
        # Testing phase starts at k=1.
        assert s.get_k() == 1
        assert s._k_star is None

    def test_testing_phase_advances_k(self):
        s = self._make_state(k_max=3, t_test=3)
        # Feed t_test steps for k=1.
        for _ in range(3):
            s.update(k_used=1, accepted=1)
        # Should now be testing k=2.
        assert s.get_k() == 2
        assert s._k_star is None

    def test_testing_phase_completes_and_selects_k_star(self):
        # With cost_factor=0 every k has the same utility, so k_max should win
        # (or k=1 if tie-breaking favours lower k — either is valid, but the
        # implementation picks the first k that strictly exceeds baseline=1).
        s = self._make_state(k_max=3, t_test=2, cost_factor=0.0)
        # Feed 2 steps for k=1 with acceptance=1 each → mean_accepted=1
        for _ in range(2):
            s.update(k_used=1, accepted=1)
        # k=2 testing
        for _ in range(2):
            s.update(k_used=2, accepted=2)
        # k=3 testing — triggers selection
        for _ in range(2):
            s.update(k_used=3, accepted=2)

        # All utilities > 1 (cost_factor=0 → utility = 1 + mean_accepted > 1).
        assert s._k_star is not None
        assert 1 <= s._k_star <= 3

    def test_disables_when_utility_below_baseline(self):
        # With cost_factor=1 and acceptance=0, utility(k)= 1/(1+k) < 1.
        s = self._make_state(k_max=2, t_test=2, cost_factor=1.0)
        for _ in range(2):
            s.update(k_used=1, accepted=0)  # utility = 1/2 = 0.5 < 1
        for _ in range(2):
            s.update(k_used=2, accepted=0)  # utility = 1/3 < 1

        # No k exceeds the baseline → k* = 0 (speculation disabled).
        assert s._k_star == 0
        assert s.get_k() == 0

    def test_selects_best_k(self):
        # k=1 is good (high acceptance), k=2 is bad (low acceptance).
        s = self._make_state(k_max=2, t_test=4, cost_factor=0.5)
        # k=1: accepted = 1 each step → mean=1, utility = 2/(1+0.5) ≈ 1.33
        for _ in range(4):
            s.update(k_used=1, accepted=1)
        # k=2: accepted = 0 each step → mean=0, utility = 1/(1+1) = 0.5
        for _ in range(4):
            s.update(k_used=2, accepted=0)

        assert s._k_star == 1

    def test_production_phase_stable(self):
        # After selection, get_k() always returns the same value.
        s = self._make_state(k_max=2, t_test=1, cost_factor=0.0)
        s.update(k_used=1, accepted=1)
        s.update(k_used=2, accepted=2)
        k_star = s._k_star
        assert k_star is not None

        # Production steps should not change k_star (no re-test).
        for _ in range(100):
            s.update(k_used=k_star, accepted=1)
        assert s._k_star == k_star

    def test_re_test_interval_triggers_new_testing(self):
        s = self._make_state(k_max=2, t_test=1, cost_factor=0.0, re_test=3)
        # Complete the testing phase quickly.
        s.update(k_used=1, accepted=1)
        s.update(k_used=2, accepted=2)
        assert s._k_star is not None
        k_star = s._k_star

        # Three production steps should trigger re-testing.
        for _ in range(3):
            s.update(k_used=k_star, accepted=1)
        assert s._k_star is None  # back in testing phase
        assert s._test_k == 1  # restarted from k=1

    def test_update_ignores_out_of_range_k(self):
        s = self._make_state(k_max=3, t_test=5)
        # k=0 and k > k_max should be no-ops.
        s.update(k_used=0, accepted=0)
        s.update(k_used=4, accepted=2)
        assert s.get_k() == 1  # still in initial state

    def test_utility_with_no_observations(self):
        s = self._make_state(k_max=3, t_test=5, cost_factor=0.5)
        # _utility for a k with no observations returns 0.
        assert s._utility(2) == 0.0


# ---------------------------------------------------------------------------
# CascadeTracker tests
# ---------------------------------------------------------------------------


class TestCascadeTracker:
    def _make_tracker(self, k_max=3, t_test=2, cost_factor=0.0, re_test=0):
        return CascadeTracker(
            k_max=k_max,
            t_test=t_test,
            cost_factor=cost_factor,
            re_test_interval=re_test,
        )

    def test_add_and_get_k(self):
        tracker = self._make_tracker(k_max=3)
        tracker.add_request("req-1")
        # Initially in testing phase with k=1.
        assert tracker.get_k("req-1") == 1

    def test_unknown_request_returns_k_max(self):
        tracker = self._make_tracker(k_max=4)
        # A request that was never added should fall back to k_max.
        assert tracker.get_k("unknown") == 4

    def test_remove_request(self):
        tracker = self._make_tracker(k_max=3)
        tracker.add_request("req-1")
        tracker.remove_request("req-1")
        # After removal, falls back to k_max for unknown request.
        assert tracker.get_k("req-1") == 3

    def test_update_propagates_to_state(self):
        tracker = self._make_tracker(k_max=2, t_test=2, cost_factor=0.0)
        tracker.add_request("req-1")

        # Two updates for k=1 moves to k=2.
        tracker.update("req-1", k_used=1, accepted=1)
        tracker.update("req-1", k_used=1, accepted=1)
        assert tracker.get_k("req-1") == 2

    def test_multiple_requests_independent(self):
        tracker = self._make_tracker(k_max=2, t_test=2, cost_factor=0.0)
        tracker.add_request("req-a")
        tracker.add_request("req-b")

        # Advance req-a through full testing.
        for _ in range(2):
            tracker.update("req-a", k_used=1, accepted=1)
        for _ in range(2):
            tracker.update("req-a", k_used=2, accepted=2)

        # req-b should still be at k=1.
        assert tracker.get_k("req-b") == 1
        assert tracker._states["req-a"]._k_star is not None

    def test_update_unknown_request_is_noop(self):
        tracker = self._make_tracker(k_max=3)
        # Should not raise.
        tracker.update("no-such-req", k_used=2, accepted=1)

    def test_remove_unknown_request_is_noop(self):
        tracker = self._make_tracker(k_max=3)
        # Should not raise.
        tracker.remove_request("ghost")


# ---------------------------------------------------------------------------
# Integration-style: simulate a full request lifecycle
# ---------------------------------------------------------------------------


def test_full_request_lifecycle():
    """Simulate a request that goes from testing → production → done."""
    k_max = 3
    t_test = 5
    tracker = CascadeTracker(k_max=k_max, t_test=t_test, cost_factor=0.1)
    req_id = "req-lifecycle"
    tracker.add_request(req_id)

    # Testing phase: k=1 with perfect acceptance.
    for _ in range(t_test):
        assert tracker.get_k(req_id) == 1
        tracker.update(req_id, k_used=1, accepted=1)

    # Testing phase: k=2 with perfect acceptance.
    for _ in range(t_test):
        assert tracker.get_k(req_id) == 2
        tracker.update(req_id, k_used=2, accepted=2)

    # Testing phase: k=3 with perfect acceptance.
    for _ in range(t_test):
        assert tracker.get_k(req_id) == 3
        tracker.update(req_id, k_used=3, accepted=3)

    # Production phase should be chosen now.
    k_star = tracker.get_k(req_id)
    assert k_star in range(1, k_max + 1), f"Unexpected k_star={k_star}"

    # Stable production steps.
    for _ in range(20):
        assert tracker.get_k(req_id) == k_star
        tracker.update(req_id, k_used=k_star, accepted=k_star - 1)

    # Cleanup.
    tracker.remove_request(req_id)
    assert req_id not in tracker._states


def test_cascade_disables_speculation_for_bad_acceptance():
    """When no k is profitable, Cascade should disable speculation (k=0)."""
    k_max = 3
    t_test = 10
    # cost_factor=1 means each draft token doubles the cost.
    tracker = CascadeTracker(k_max=k_max, t_test=t_test, cost_factor=1.0)
    req_id = "req-bad"
    tracker.add_request(req_id)

    # Feed 0 accepted tokens for every k → all utilities < 1.
    for k in range(1, k_max + 1):
        for _ in range(t_test):
            tracker.update(req_id, k_used=k, accepted=0)

    assert tracker.get_k(req_id) == 0  # speculation disabled


# ---------------------------------------------------------------------------
# Scheduler integration tests
# ---------------------------------------------------------------------------


class TestCascadeSchedulerIntegration:
    """Tests that the Scheduler wires Cascade correctly end-to-end."""

    def test_cascade_tracker_created_when_enabled(self):
        scheduler = _make_cascade_scheduler(num_spec_tokens=4)
        assert scheduler.cascade_tracker is not None
        assert scheduler.cascade_tracker.k_max == 4

    def test_cascade_tracker_absent_when_disabled(self):
        """Without enable_cascade, cascade_tracker must be None (no overhead)."""
        from tests.v1.core.utils import create_scheduler

        scheduler = create_scheduler(num_speculative_tokens=4)
        assert scheduler.cascade_tracker is None

    def test_lookahead_extra_is_zero_for_ngram(self):
        """ngram has num_lookahead_tokens=0, so _cascade_lookahead returns 0."""
        scheduler = _make_cascade_scheduler(num_spec_tokens=4)
        # ngram does not use KV cache for drafting, so _cascade_lookahead must
        # always return 0 regardless of the cascade k.
        scheduler.cascade_tracker.add_request("req-ngram")
        assert scheduler._cascade_lookahead("req-ngram") == 0

    def test_cascade_lookahead_fallback_without_tracker(self):
        """_cascade_lookahead returns num_lookahead_tokens when tracker is None."""
        from tests.v1.core.utils import create_scheduler

        scheduler = create_scheduler(num_speculative_tokens=4)
        # Manually ensure tracker is None (already the case without cascade).
        assert scheduler.cascade_tracker is None
        assert scheduler._cascade_lookahead("any-req") == scheduler.num_lookahead_tokens

    def test_cascade_lookahead_testing_phase_starts_at_one(self):
        """Immediately after add_request, cascade k=1.

        For ngram (num_lookahead_tokens=0), _cascade_lookahead returns 0 since
        ngram does not use KV cache for draft tokens.  The k value is still 1
        as reported by the tracker — cascade_tracker.get_k returns the correct k.
        """
        scheduler = _make_cascade_scheduler(num_spec_tokens=4)
        scheduler.cascade_tracker.add_request("req-a")
        # ngram: lookahead is always 0 (no KV cache for drafts)
        assert scheduler._cascade_lookahead("req-a") == 0
        # but the cascade k starts at 1
        assert scheduler.cascade_tracker.get_k("req-a") == 1

    def test_cascade_lookahead_zero_when_speculation_disabled(self):
        """When cascade picks k=0, lookahead must be 0 (no slots wasted)."""
        scheduler = _make_cascade_scheduler(num_spec_tokens=4, cost_factor=1.0)
        scheduler.cascade_tracker.add_request("req-bad")
        # Drive acceptance to 0 for all k → utility < 1 → k*=0.
        t = scheduler.cascade_tracker.t_test
        for k in range(1, 5):
            for _ in range(t):
                scheduler.cascade_tracker.update("req-bad", k_used=k, accepted=0)
        assert scheduler.cascade_tracker.get_k("req-bad") == 0
        assert scheduler._cascade_lookahead("req-bad") == 0

    def test_cascade_lookahead_unknown_request_returns_k_max(self):
        """Unknown req_id: tracker returns k_max; ngram always yields 0 lookahead."""
        scheduler = _make_cascade_scheduler(num_spec_tokens=4)
        # Tracker correctly returns k_max for unknown requests.
        assert scheduler.cascade_tracker.get_k("unknown-req") == 4
        # But ngram (num_lookahead_tokens=0) means _cascade_lookahead returns 0.
        assert scheduler._cascade_lookahead("unknown-req") == 0

    def test_add_and_remove_request_lifecycle(self):
        """add_request creates cascade state; _free_request removes it."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request, RequestStatus

        scheduler = _make_cascade_scheduler(num_spec_tokens=4, t_test=2)

        req = Request(
            request_id="req-lifecycle",
            prompt_token_ids=list(range(10)),
            sampling_params=SamplingParams(max_tokens=20),
            pooling_params=None,
        )
        scheduler.add_request(req)
        assert "req-lifecycle" in scheduler.cascade_tracker._states

        req.status = RequestStatus.FINISHED_ABORTED
        scheduler._free_request(req)
        assert "req-lifecycle" not in scheduler.cascade_tracker._states
