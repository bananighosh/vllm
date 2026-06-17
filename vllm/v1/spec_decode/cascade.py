# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cascade: Utility-Driven Adaptive k for MoE Speculative Decoding.

Reference: https://arxiv.org/abs/2506.20675

On MoE models every additional draft token activates a different expert
subset, so the target-model verification cost scales roughly linearly with k
(the number of speculative tokens).  A static k therefore incurs unnecessary
overhead when the acceptance rate is low.

Cascade fixes this by choosing, *per request*, the k that maximises:

    utility(k) = (1 + E[accepted | k]) / (1 + k * cost_factor)

where cost_factor ∈ [0, 1] is the per-draft-token verification overhead
relative to a single-token forward pass (≈ 0 for dense, ≈ 1 for pure-MoE).

The algorithm has two phases:
  Testing   – cycle through k ∈ {1, …, k_max} for t_test steps each,
              accumulating acceptance statistics.
  Production – lock in k* = argmax_k utility(k).  If utility(k*) ≤ 1
              (no k beats greedy decoding), speculation is disabled (k=0)
              for this request without evicting the draft model.

Optionally a re_test_interval > 0 triggers a fresh testing phase every
re_test_interval production steps so the system can adapt to changing
context characteristics.
"""

from dataclasses import dataclass, field


@dataclass
class RequestCascadeState:
    """Per-request Cascade state."""

    k_max: int
    t_test: int
    cost_factor: float
    re_test_interval: int

    # --- testing-phase state ---
    _test_k: int = field(default=1, init=False)
    _test_step: int = field(default=0, init=False)

    # Per-k accumulators indexed by k (index 0 unused).
    _accepted_sum: list[float] = field(default_factory=list, init=False)
    _step_count: list[int] = field(default_factory=list, init=False)

    # --- production-phase state ---
    # None  → still in testing phase.
    # 0     → speculation disabled for this request.
    # 1..k_max → optimal k chosen.
    _k_star: int | None = field(default=None, init=False)
    _prod_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._accepted_sum = [0.0] * (self.k_max + 1)
        self._step_count = [0] * (self.k_max + 1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_k(self) -> int:
        """Return the number of spec tokens to request for this step."""
        if self._k_star is None:
            return self._test_k
        return self._k_star

    def update(self, k_used: int, accepted: int) -> None:
        """Record the outcome of one speculative step.

        Args:
            k_used:   Number of draft tokens that were scheduled.
            accepted: Number of those draft tokens that were accepted.
        """
        if k_used <= 0 or k_used > self.k_max:
            return

        self._accepted_sum[k_used] += accepted
        self._step_count[k_used] += 1

        if self._k_star is None:
            self._advance_testing_phase()
        else:
            self._advance_production_phase()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _utility(self, k: int) -> float:
        """Expected-speedup utility for k speculative tokens."""
        if self._step_count[k] == 0:
            return 0.0
        mean_accepted = self._accepted_sum[k] / self._step_count[k]
        return (1.0 + mean_accepted) / (1.0 + k * self.cost_factor)

    def _advance_testing_phase(self) -> None:
        self._test_step += 1
        if self._test_step >= self.t_test:
            self._test_k += 1
            self._test_step = 0
            if self._test_k > self.k_max:
                self._select_k_star()

    def _advance_production_phase(self) -> None:
        if self.re_test_interval <= 0:
            return
        self._prod_steps += 1
        if self._prod_steps >= self.re_test_interval:
            # Re-enter testing phase with a clean slate.
            self._test_k = 1
            self._test_step = 0
            self._k_star = None
            self._prod_steps = 0
            self._accepted_sum = [0.0] * (self.k_max + 1)
            self._step_count = [0] * (self.k_max + 1)

    def _select_k_star(self) -> None:
        """Pick k* after the testing phase completes."""
        best_k = 0
        best_u = 1.0  # baseline: no speculation (utility = 1.0)
        for k in range(1, self.k_max + 1):
            u = self._utility(k)
            if u > best_u:
                best_u = u
                best_k = k
        self._k_star = best_k
        self._prod_steps = 0


class CascadeTracker:
    """Manages per-request Cascade states for the scheduler.

    Args:
        k_max:             Maximum speculative tokens (K_max in the paper).
        t_test:            Steps to test each k value during the testing phase.
        cost_factor:       Per-draft-token verification overhead fraction.
                           Use ~0 for dense models, ~1 for pure-MoE.
        re_test_interval:  Steps between re-testing phases (0 = never).
    """

    def __init__(
        self,
        k_max: int,
        t_test: int = 20,
        cost_factor: float = 0.5,
        re_test_interval: int = 0,
    ) -> None:
        self.k_max = k_max
        self.t_test = t_test
        self.cost_factor = cost_factor
        self.re_test_interval = re_test_interval
        self._states: dict[str, RequestCascadeState] = {}

    def add_request(self, req_id: str) -> None:
        self._states[req_id] = RequestCascadeState(
            k_max=self.k_max,
            t_test=self.t_test,
            cost_factor=self.cost_factor,
            re_test_interval=self.re_test_interval,
        )

    def remove_request(self, req_id: str) -> None:
        self._states.pop(req_id, None)

    def get_k(self, req_id: str) -> int:
        """Return the recommended number of spec tokens for *req_id*."""
        state = self._states.get(req_id)
        return state.get_k() if state is not None else self.k_max

    def update(self, req_id: str, k_used: int, accepted: int) -> None:
        """Record the acceptance outcome for *req_id*."""
        state = self._states.get(req_id)
        if state is not None:
            state.update(k_used, accepted)
