"""Stage 5b — Monte-Carlo liability cashflow paths.

Generates ``n_paths`` independent liability ``CashFlowStream`` realisations
under binomial cohort transitions (Maffra/Armstrong/Pennanen 2021 Eq. 1)
applied uniformly across cohorts. Path independence is guaranteed via
``numpy.random.SeedSequence(seed).spawn(n_paths)``: each path gets its own
spawned ``Generator``, and inside each simulator the per-cohort RNGs are
further spawned from that path's master rng (see
``ClosedFundSimulator._get_cohort_rng``).

This is a pure-Python loop: no network calls, no ACTUS service round-trips.
The bottleneck is the per-path simulator pass over all cohorts and years.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...models.cashFlowStream import CashFlowStream
from ..entry import EntryPolicy
from ..fund import PensionFund
from ..mortality import MortalityTable
from .Stage4_dynamic import DynamicFundSimulator, Stage4Dynamics


def simulate_liability_paths(
    fund: PensionFund,
    entry_policy: EntryPolicy,
    dynamics: Stage4Dynamics,
    mortality: MortalityTable,
    *,
    horizon_years: int,
    n_paths: int,
    seed: Optional[int] = None,
    mortality_filter: str = "all",
) -> List[CashFlowStream]:
    """Return ``n_paths`` independent stochastic-mortality liability streams.

    Parameters
    ----------
    fund, entry_policy, dynamics, mortality
        Same building blocks as a standard ``DynamicFundSimulator`` run.
    horizon_years
        Simulation horizon in years.
    n_paths
        Number of Monte-Carlo paths to generate.
    seed
        Master seed. ``None`` uses non-reproducible entropy.
    mortality_filter
        ``"all"`` (default): binomial decrement applied to active AND
        retired cohorts each year. ``"retirees"``: binomial decrement
        only on retired cohorts (matches the Stage-3 deterministic scope
        but with stochastic draws); actives remain at fixed headcount.

    Returns
    -------
    List[CashFlowStream]
        Length ``n_paths``. Element ``i`` is the liability stream of
        path ``i`` and is paired position-wise with stochastic asset
        streams in ``StochasticALMAnalysis`` / ``RiskAttribution``.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    seed_seq = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in seed_seq.spawn(int(n_paths))]
    out: List[CashFlowStream] = []
    for rng in rngs:
        sim = DynamicFundSimulator(
            fund=fund,
            entry_policy=entry_policy,
            dynamics=dynamics,
            mortality=mortality,
            stochastic_mortality=True,
            mortality_filter=mortality_filter,
            rng=rng,
        )
        out.append(sim.run(horizon_years=horizon_years))
    return out
