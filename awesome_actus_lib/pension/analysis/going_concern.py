"""Stage 5d (Baustein 1) — going-concern solvency funding ratio.

Spec: ``docs/stage5d_going_concern_spec.md``.

Verified hook point (spec asked for this check before coding): Stage 5c's
``StochasticFundingRatioAnalysis._path_V`` does NOT run a sequential annual
roll — it books V(as_of) in ONE SHOT from the t0 state: frozen scalar book
values for bonds/cash, cumulative ACTUS "IP" coupon sums, cumulative net
liability flows at 0% return, and a single scalar equity sleeve scaled by
the GBM price relative S_t/S_0. There are no buckets and no in-loop DG_t.

Consequently the sequential per-path annual roll the spec relies on is
introduced HERE, in the subclass: with a ``RebalancingPolicy`` the asset
side is rolled forward over base-date anniversaries with two book-value
buckets (BONDS = fixed income incl. cash, EQUITY), reinvestment of net
flows, and fixed-mix rebalancing. DG_t is available in-loop in this roll,
which is what makes the Baustein-3 Sanierungs-trigger architecturally
compatible later (asset-only inflow, no liability events touched).

Regression principle (spec, Akzeptanz-Assert 1): with ``rebalancing=None``
and ``sanierung=None`` every result is byte-identical to Stage 5c — the
``_path_V`` override delegates verbatim to the parent. Run the built-in
offline regression check (no ACTUS service calls) with:

    python3 -m awesome_actus_lib.pension.analysis.going_concern

Baustein 3 (``SanierungsPolicy``): DG-conditional Sanierungsbeitraege per
Art. 65d BVG as a PURE ASSET INFLOW — SB_t = sb_factor * SAV_CONTRIB_t,
active in year t iff DG_{t-1} < trigger_dg (hysteresis via exit_dg,
optional max_years cap per path). SB events are logged in a SEPARATE
gc-event log per path (``gc_events``, type ``SANIERUNG_SB``) and are NOT
written into the liability stream — provenance stays clean, and because
no liability events are created or changed, the measure is compatible
with the generate-then-pair architecture (spec Architektur-Regel).
SB ∝ SAV_CONTRIB is the documented payroll proxy (Lohnsumme is not in
the event stream); ``sb_factor`` is a free parameter.

Documented simplifications (spec Limitations):
  * EQUITY accrues the book-value drift ``equity_return`` (mark-to-model)
    instead of the 5c GBM mark-to-market. With ``equity_sigma > 0`` the
    annual gross return becomes lognormal per path and anniversary year,
    (1+r) * exp(sigma*Z - sigma^2/2) with Z ~ N(0,1), so its expectation
    stays (1+r) — sigma is a mean-preserving spread around the drift.
    Shocks are seeded via ``np.random.SeedSequence(seed).spawn(n_paths)``
    (one generator per path, same pattern as the 5b liability simulator);
    shock k on path i is the k-th draw of generator i, so values are
    independent of the order in which valuation dates are requested.
    Final partial windows (non-anniversary ``as_of``) accrue at the
    deterministic drift only.
  * The cash book is folded into the BONDS bucket: the ACTUS event stream
    does not separate fixed-bond from cash-account "IP" income without a
    contractId->bucket mapping, and the spec's scenarios use two buckets.
  * BONDS income: the ORIGINAL book value earns exactly the ACTUS "IP"
    coupons of the original contracts (run-off semantics, like 5c).
    Rebalanced/reinvested money on top of that earns nothing by default —
    no new ACTUS contracts are generated. ``bond_yield`` (Baustein 2c)
    fixes this: the incremental BONDS balance ABOVE the initial book value
    accrues a flat book yield per year (simple within the window, positive
    part only). ``bond_yield=None`` is byte-identical to the
    no-yield behaviour (same regression pattern as ``equity_sigma=0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..events import SAV_CONTRIB
from ..fund import PensionFund
from ..mortality import MortalityTable
from ..simulators.liability_paths import LiabilityPath
from .stochastic_alm import StochasticALMAnalysis
from .stochastic_funding_ratio import StochasticFundingRatioAnalysis

BONDS = "BONDS"
EQUITY = "EQUITY"
_BUCKETS = (BONDS, EQUITY)


def _validate_weights(weights: Dict[str, float], name: str) -> None:
    if set(weights.keys()) != set(_BUCKETS):
        raise ValueError(
            f"{name} keys must be exactly {set(_BUCKETS)}, got {set(weights.keys())}"
        )
    for k, w in weights.items():
        if not 0.0 <= float(w) <= 1.0:
            raise ValueError(f"{name}[{k!r}] must be in [0, 1], got {w}")
    total = sum(float(w) for w in weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to 1, got {total}")


@dataclass(frozen=True)
class RebalancingPolicy:
    """Reinvestment + fixed-mix rebalancing onto target weights (Stufe 1).

    Parameters
    ----------
    target_weights:
        e.g. ``{"BONDS": 0.40, "EQUITY": 0.60}``; must sum to 1.
    frequency_years:
        Rebalance every N base-date anniversaries. Default 1 = yearly.
    cost_bps:
        Optional proportional cost in basis points on one-sided turnover,
        deducted pro rata to the target weights (so post-cost weights stay
        exactly at target). Default 0 = kostenlos.
    shift_at_year / shift_weights:
        Optional allocation shift: from anniversary ``k >= shift_at_year``
        on, ``shift_weights`` replace ``target_weights`` (Szenario C).
    """

    target_weights: Dict[str, float]
    frequency_years: int = 1
    cost_bps: float = 0.0
    shift_at_year: Optional[int] = None
    shift_weights: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        _validate_weights(self.target_weights, "target_weights")
        if int(self.frequency_years) < 1:
            raise ValueError(
                f"frequency_years must be >= 1, got {self.frequency_years}"
            )
        if self.cost_bps < 0.0:
            raise ValueError(f"cost_bps must be >= 0, got {self.cost_bps}")
        if (self.shift_at_year is None) != (self.shift_weights is None):
            raise ValueError(
                "shift_at_year and shift_weights must be given together"
            )
        if self.shift_weights is not None:
            _validate_weights(self.shift_weights, "shift_weights")
            if int(self.shift_at_year) < 1:
                raise ValueError(
                    f"shift_at_year must be >= 1, got {self.shift_at_year}"
                )

    def effective_weights(self, year_index: int) -> Dict[str, float]:
        """Target weights in force at base-date anniversary ``year_index``."""
        if self.shift_at_year is not None and year_index >= self.shift_at_year:
            return self.shift_weights
        return self.target_weights


@dataclass(frozen=True)
class SanierungsPolicy:
    """DG-conditional Sanierungsbeitrag (Art. 65d BVG) — pure asset inflow.

    Active in year t iff DG_{t-1} < ``trigger_dg``; once active it stays
    active until DG_{t-1} >= ``exit_dg`` (hysteresis, ``exit_dg >=
    trigger_dg``; equal values = memoryless trigger). When active,
    ``SB_t = sb_factor * SAV_CONTRIB_t`` (the path's savings contributions
    of the year — documented payroll proxy) is booked as an asset inflow
    and logged as ``SANIERUNG_SB`` in the separate gc-event log. The
    liability stream is never touched. ``max_years`` caps the number of
    SB-paying years per path (cumulative, counted when SB > 0 flows).
    """

    trigger_dg: float = 1.00
    exit_dg: float = 1.00
    sb_factor: float = 0.5
    max_years: Optional[int] = None

    def __post_init__(self) -> None:
        if self.exit_dg < self.trigger_dg:
            raise ValueError(
                f"exit_dg ({self.exit_dg}) must be >= trigger_dg "
                f"({self.trigger_dg}) — hysteresis exit cannot lie below entry"
            )
        if self.sb_factor < 0.0:
            raise ValueError(f"sb_factor must be >= 0, got {self.sb_factor}")
        if self.max_years is not None and int(self.max_years) < 1:
            raise ValueError(f"max_years must be >= 1, got {self.max_years}")


class GoingConcernFundingRatioAnalysis(StochasticFundingRatioAnalysis):
    """Stage-5c solvency funding ratio with opt-in going-concern asset roll.

    With ``rebalancing=None`` and ``sanierung=None`` every result is
    byte-identical to ``StochasticFundingRatioAnalysis`` (Stage 5c): the
    overridden ``_path_V`` delegates straight to the parent and nothing
    else in the valuation chain is touched.

    With a ``RebalancingPolicy``, V_t,i is produced by a sequential annual
    roll per path (order per year, spec section "Per-Pfad-Roll"):

      1. accrue per bucket — BONDS gets the period's ACTUS "IP" coupons
         (per-path events, same filter as 5c) plus, with ``bond_yield``,
         a flat book yield on the incremental balance above the initial
         book value; EQUITY grows at the book-value drift
         ``equity_return`` (with ``equity_sigma > 0``: seeded lognormal
         annual returns per path, see module docstring);
      2. net liability cashflows of the paired ``LiabilityPath`` (excl.
         ``RISK_CONTRIB`` — Mai-Fix nicht regressieren) are added/withdrawn
         proportionally to the current weights;
      3. Sanierung (Baustein 3): if DG_{t-1} < trigger (hysteresis via
         exit_dg, max_years cap), ``SB_t = sb_factor * SAV_CONTRIB_t`` is
         booked as a pure asset inflow and logged via ``gc_events`` —
         never into the liability stream;
      4. rebalance onto the effective target weights every
         ``frequency_years`` anniversaries (book transfer, optional
         ``cost_bps``; from ``shift_at_year`` on, ``shift_weights`` apply);
      5. DG_t is available in-loop (exposed via ``weight_path``).

    The liability side (PC_t,i) is untouched — Baustein 1 changes only how
    V_t,i evolves.
    """

    def __init__(
        self,
        salm: StochasticALMAnalysis,
        fund: PensionFund,
        mortality: MortalityTable,
        *,
        bond_book: float,
        cash_book: float,
        equity_sleeve_0: float,
        liab_paths: Optional[List[LiabilityPath]] = None,
        rebalancing: Optional[RebalancingPolicy] = None,
        sanierung=None,
        equity_return: float = 0.03,
        equity_sigma: float = 0.0,
        equity_seed: Optional[int] = None,
        bond_yield: Optional[float] = None,
        initial_weights: Optional[Dict[str, float]] = None,
    ):
        if sanierung is not None and not isinstance(sanierung, SanierungsPolicy):
            raise TypeError(
                f"sanierung must be a SanierungsPolicy or None, "
                f"got {type(sanierung).__name__}"
            )
        if sanierung is not None and rebalancing is None:
            raise ValueError(
                "sanierung requires an active RebalancingPolicy; with "
                "rebalancing=None the 5c delegation runs and would silently "
                "ignore the SanierungsPolicy."
            )
        if rebalancing is not None and liab_paths is None:
            raise ValueError(
                "rebalancing requires forward mode (liab_paths is not None); "
                "the legacy V_i(d)/PC_t0 mode has no per-path roll to rebalance."
            )
        if equity_sigma < 0.0:
            raise ValueError(f"equity_sigma must be >= 0, got {equity_sigma}")
        if equity_sigma > 0.0 and rebalancing is None:
            raise ValueError(
                "equity_sigma > 0 requires an active RebalancingPolicy; "
                "with rebalancing=None the 5c delegation ignores it, which "
                "would silently drop the requested equity volatility."
            )
        if bond_yield is not None and rebalancing is None:
            raise ValueError(
                "bond_yield requires an active RebalancingPolicy; with "
                "rebalancing=None there is no incremental BONDS balance "
                "to accrue it on."
            )
        if initial_weights is not None:
            _validate_weights(initial_weights, "initial_weights")
        super().__init__(
            salm,
            fund,
            mortality,
            bond_book=bond_book,
            cash_book=cash_book,
            equity_sleeve_0=equity_sleeve_0,
            liab_paths=liab_paths,
        )
        self.rebalancing = rebalancing
        self.sanierung = sanierung
        self.equity_return = float(equity_return)
        self.equity_sigma = float(equity_sigma)
        self.equity_seed = equity_seed
        self.bond_yield = None if bond_yield is None else float(bond_yield)
        self.initial_weights = initial_weights

        # Per-path flow arrays parsed once (the roll re-reads them per
        # window; parsing the event DataFrames every time is O(minutes)
        # at quickstart scale).
        self._flow_cache: Dict[int, Tuple[np.ndarray, ...]] = {}
        # PC_t,i is pure in (date, path) — cached because the Sanierungs-
        # trigger needs DG_{t-1} in every roll window of every as_of call.
        self._pc_cache: Dict[Tuple[int, str], float] = {}

        # Lognormal equity shocks: one generator per path via
        # SeedSequence.spawn (5b/5c pattern); shock k = k-th draw of the
        # path's generator, cached so valuation-date order cannot matter.
        if self.equity_sigma > 0.0:
            seed = equity_seed if equity_seed is not None else getattr(salm, "seed", None)
            seed_seq = np.random.SeedSequence(seed)
            self._eq_rngs = [
                np.random.default_rng(s) for s in seed_seq.spawn(int(salm.n_paths))
            ]
        else:
            self._eq_rngs = None
        self._eq_shock_cache: Dict[int, List[float]] = {}

    # ------------------------------------------------------------------ #
    # asset-side valuation                                               #
    # ------------------------------------------------------------------ #

    def _path_V(self, as_of: str, path_i: int) -> float:
        # Akzeptanz-Assert 1: without a policy this IS the Stage-5c
        # one-shot booking, byte for byte.
        if self.rebalancing is None:
            return super()._path_V(as_of, path_i)
        v, _ = self._roll_rebalanced(as_of, path_i, collect_log=False)
        return v

    def weight_path(self, path_i: int, as_of: str) -> pd.DataFrame:
        """Annual roll log for one path up to ``as_of``.

        Columns: date, year_index (anniversary k, NaN for a final partial
        window), w_BONDS / w_EQUITY (post-rebalance), V, dg, rebalanced,
        target_BONDS / target_EQUITY (effective targets, NaN off-cycle).
        Serves Akzeptanz-Asserts 3 und 5 and the Gewichtspfad plots.
        """
        if self.rebalancing is None:
            raise ValueError("weight_path requires an active RebalancingPolicy")
        _, log = self._roll_rebalanced(as_of, path_i, collect_log=True)
        return pd.DataFrame(log)

    def gc_events(self, path_i: int, as_of: str) -> pd.DataFrame:
        """Separate gc-event log for one path up to ``as_of``.

        Contains the Sanierungsbeitraege as ``SANIERUNG_SB`` rows (time,
        type, payoff, dg_prev, path) — deliberately NOT part of the
        liability stream, so liability provenance stays clean. Empty
        DataFrame when no SanierungsPolicy is set or none fired.
        """
        if self.rebalancing is None:
            raise ValueError("gc_events requires an active RebalancingPolicy")
        _, log = self._roll_rebalanced(as_of, path_i, collect_log=True)
        rows = [
            {"time": r["date"], "type": "SANIERUNG_SB", "payoff": r["sb"],
             "dg_prev": r["dg_prev"], "path": path_i}
            for r in log if r["sb"] > 0.0
        ]
        return pd.DataFrame(rows, columns=["time", "type", "payoff", "dg_prev", "path"])

    def _bucket_init(self) -> Dict[str, float]:
        total0 = self.bond_book + self.cash_book + self.equity_sleeve_0
        if self.initial_weights is None:
            # Cash folded into the fixed-income bucket (see module docstring).
            return {BONDS: self.bond_book + self.cash_book,
                    EQUITY: self.equity_sleeve_0}
        return {b: self.initial_weights[b] * total0 for b in _BUCKETS}

    def _path_flows(
        self, path_i: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pre-filtered (times, payoffs) arrays per path, parsed once.

        Asset side keeps only "IP" events, liability side drops
        RISK_CONTRIB — exactly the 5c filters, just cached as numpy arrays
        so the per-window sums of the roll stay cheap.
        """
        cached = self._flow_cache.get(path_i)
        if cached is not None:
            return cached

        asset_df = self.salm._asset_cfs[path_i].events_df
        if asset_df is None or asset_df.empty:
            a_times = np.empty(0, dtype="datetime64[ns]")
            a_pay = np.empty(0, dtype=float)
        else:
            ip = asset_df[asset_df["type"] == "IP"]
            a_times = pd.to_datetime(ip["time"]).to_numpy(dtype="datetime64[ns]")
            a_pay = ip["payoff"].to_numpy(dtype=float)

        liab_df = self.liab_paths[path_i].cashflows.events_df
        if liab_df is None or liab_df.empty:
            l_times = np.empty(0, dtype="datetime64[ns]")
            l_pay = np.empty(0, dtype=float)
            s_times = np.empty(0, dtype="datetime64[ns]")
            s_pay = np.empty(0, dtype=float)
        else:
            nl = liab_df[liab_df["type"] != "RISK_CONTRIB"]
            l_times = pd.to_datetime(nl["time"]).to_numpy(dtype="datetime64[ns]")
            l_pay = nl["payoff"].to_numpy(dtype=float)
            sav = liab_df[liab_df["type"] == SAV_CONTRIB]
            s_times = pd.to_datetime(sav["time"]).to_numpy(dtype="datetime64[ns]")
            s_pay = sav["payoff"].to_numpy(dtype=float)

        cached = (a_times, a_pay, l_times, l_pay, s_times, s_pay)
        self._flow_cache[path_i] = cached
        return cached

    def _window_bond_income(
        self, path_i: int, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> float:
        """ACTUS "IP" coupons in (start_ts, end_ts] — same filter as 5c."""
        a_times, a_pay, _, _, _, _ = self._path_flows(path_i)
        if a_times.size == 0:
            return 0.0
        mask = (a_times > start_ts.to_datetime64()) & (a_times <= end_ts.to_datetime64())
        return float(a_pay[mask].sum())

    def _window_net_liab(
        self, path_i: int, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> float:
        """Net liability flows in (start_ts, end_ts], excl. RISK_CONTRIB."""
        _, _, l_times, l_pay, _, _ = self._path_flows(path_i)
        if l_times.size == 0:
            return 0.0
        mask = (l_times > start_ts.to_datetime64()) & (l_times <= end_ts.to_datetime64())
        return float(l_pay[mask].sum())

    def _window_sav_contrib(
        self, path_i: int, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> float:
        """SAV_CONTRIB flows in (start_ts, end_ts] — SB assessment base."""
        _, _, _, _, s_times, s_pay = self._path_flows(path_i)
        if s_times.size == 0:
            return 0.0
        mask = (s_times > start_ts.to_datetime64()) & (s_times <= end_ts.to_datetime64())
        return float(s_pay[mask].sum())

    def _path_PC_cached(self, as_of: str, path_i: int) -> float:
        key = (path_i, as_of)
        pc = self._pc_cache.get(key)
        if pc is None:
            pc = self._path_PC(as_of, path_i)
            self._pc_cache[key] = pc
        return pc

    def _equity_growth(self, path_i: int, k_cur: Optional[int], year_frac: float) -> float:
        """Gross equity return for one roll window.

        equity_sigma == 0: deterministic drift (1+r)^year_frac — bitwise
        the Baustein-1 behaviour. equity_sigma > 0: lognormal annual gross
        return (1+r)*exp(sigma*Z - sigma^2/2) on anniversary windows
        (E[gross] = 1+r); final partial windows stay at the drift.
        """
        if self._eq_rngs is None or k_cur is None:
            return (1.0 + self.equity_return) ** year_frac
        z = self._equity_shock(path_i, k_cur)
        return (1.0 + self.equity_return) * float(
            np.exp(self.equity_sigma * z - 0.5 * self.equity_sigma ** 2)
        )

    def _equity_shock(self, path_i: int, k: int) -> float:
        """Standard-normal shock for anniversary k on path_i (1-based)."""
        shocks = self._eq_shock_cache.setdefault(path_i, [])
        rng = self._eq_rngs[path_i]
        while len(shocks) < k:
            shocks.append(float(rng.standard_normal()))
        return shocks[k - 1]

    def _roll_rebalanced(
        self, as_of: str, path_i: int, collect_log: bool
    ) -> Tuple[float, List[dict]]:
        pol = self.rebalancing
        base_ts = pd.to_datetime(self.salm.base_date)
        as_of_ts = pd.to_datetime(as_of)
        if as_of_ts <= base_ts:
            return super()._path_V(as_of, path_i), []

        # Roll grid: base-date anniversaries (index k) up to as_of, plus
        # as_of itself as a final partial window if it is no anniversary.
        # Only anniversaries can trade.
        grid: List[Tuple[pd.Timestamp, Optional[int]]] = []
        k = 1
        while True:
            t_k = base_ts + pd.DateOffset(years=k)
            if t_k > as_of_ts:
                break
            grid.append((t_k, k))
            k += 1
        if not grid or grid[-1][0] != as_of_ts:
            grid.append((as_of_ts, None))

        buckets = self._bucket_init()
        bonds_book_0 = buckets[BONDS]
        log: List[dict] = []
        t_prev = base_ts

        san = self.sanierung
        san_active = False
        sb_years_used = 0

        for t_cur, k_cur in grid:
            # Sanierungs-trigger state BEFORE this year's accruals: DG_{t-1}
            # from the post-everything state of the previous anniversary.
            # Partial final windows never pay SB (no full contribution year).
            dg_prev = float("nan")
            pay_sb = False
            if san is not None and k_cur is not None:
                v_prev = buckets[BONDS] + buckets[EQUITY]
                pc_prev = self._path_PC_cached(t_prev.strftime("%Y-%m-%d"), path_i)
                dg_prev = v_prev / pc_prev if pc_prev > 0.0 else float("nan")
                if san_active:
                    # Hysteresis: stay active until DG_{t-1} >= exit_dg.
                    # NaN (PC=0 edge) deactivates.
                    san_active = dg_prev < san.exit_dg
                else:
                    san_active = dg_prev < san.trigger_dg
                pay_sb = san_active and (
                    san.max_years is None or sb_years_used < int(san.max_years)
                )

            # 1. accrue per bucket
            year_frac = 1.0 if k_cur is not None else (
                (t_cur - t_prev).days / 365.25
            )
            buckets[EQUITY] *= self._equity_growth(path_i, k_cur, year_frac)
            if self.bond_yield is not None:
                # Baustein 2c: flat book yield on the incremental BONDS
                # balance above the initial book value, on the position
                # held over the window (start-of-window balance). The
                # original book value keeps earning the ACTUS coupons only.
                incremental = buckets[BONDS] - bonds_book_0
                if incremental > 0.0:
                    buckets[BONDS] += self.bond_yield * incremental * year_frac
            buckets[BONDS] += self._window_bond_income(path_i, t_prev, t_cur)

            # 2. net liability flows, proportional to current weights
            net_liab = self._window_net_liab(path_i, t_prev, t_cur)
            total = buckets[BONDS] + buckets[EQUITY]
            if total > 0.0:
                for b in _BUCKETS:
                    buckets[b] += net_liab * (buckets[b] / total)
            else:
                # Insolvent on this path — weights ill-defined, park in BONDS.
                buckets[BONDS] += net_liab

            # 3. Sanierung (Baustein 3): SB_t = sb_factor * SAV_CONTRIB_t as
            # pure asset inflow, distributed like the other flows. The
            # liability stream is NOT touched; logging goes to the separate
            # gc-event log (see gc_events).
            sb_amount = 0.0
            if pay_sb:
                sav_t = self._window_sav_contrib(path_i, t_prev, t_cur)
                sb_amount = san.sb_factor * sav_t
                if sb_amount > 0.0:
                    total = buckets[BONDS] + buckets[EQUITY]
                    if total > 0.0:
                        for b in _BUCKETS:
                            buckets[b] += sb_amount * (buckets[b] / total)
                    else:
                        buckets[BONDS] += sb_amount
                    sb_years_used += 1

            # 4. rebalance onto effective targets every frequency_years
            rebalanced = False
            targets: Optional[Dict[str, float]] = None
            if k_cur is not None and k_cur % int(pol.frequency_years) == 0:
                targets = pol.effective_weights(k_cur)
                total = buckets[BONDS] + buckets[EQUITY]
                if total > 0.0:
                    turnover = 0.5 * sum(
                        abs(targets[b] * total - buckets[b]) for b in _BUCKETS
                    )
                    cost = turnover * pol.cost_bps / 10_000.0
                    for b in _BUCKETS:
                        # Cost pro rata to targets keeps post-cost weights
                        # exactly at target (Akzeptanz-Assert 3).
                        buckets[b] = targets[b] * total - cost * targets[b]
                    rebalanced = True

            # 5. DG_t available in-loop (logged on demand)
            if collect_log:
                total = buckets[BONDS] + buckets[EQUITY]
                pc = self._path_PC_cached(t_cur.strftime("%Y-%m-%d"), path_i)
                log.append({
                    "date": t_cur,
                    "year_index": k_cur,
                    "w_BONDS": buckets[BONDS] / total if total > 0 else float("nan"),
                    "w_EQUITY": buckets[EQUITY] / total if total > 0 else float("nan"),
                    "V": total,
                    "dg": total / pc if pc > 0 else float("nan"),
                    "rebalanced": rebalanced,
                    "target_BONDS": targets[BONDS] if targets else float("nan"),
                    "target_EQUITY": targets[EQUITY] if targets else float("nan"),
                    "sanierung_active": san_active if san is not None else False,
                    "dg_prev": dg_prev,
                    "sb": sb_amount,
                })
            t_prev = t_cur

        return buckets[BONDS] + buckets[EQUITY], log


# ---------------------------------------------------------------------- #
# Regression check: rebalancing=None, sanierung=None  ==  Stage 5c        #
# ---------------------------------------------------------------------- #

def _regression_check_against_5c() -> None:
    """Offline acceptance asserts (synthetic fixtures, no ACTUS service).

    Covers the Baustein-1 subset of the spec's Akzeptanz-Asserts:
    #1 (byte-identical regression), #3 (post-rebalance weights at target),
    #5 (shift mechanics), plus sanierung/validation guards.
    """
    from types import SimpleNamespace

    import numpy as np

    from ..cohort import Cohort
    from ..mortality import ek2001_2005
    from ..policy import PensionPolicy

    n_paths = 4
    horizon_years = 10
    base_date = "2025-01-01"
    start_year = 2025

    policy = PensionPolicy()
    fund = PensionFund(policy=policy, start_date=f"{base_date}T00:00:00")
    fund.add_cohort(Cohort(
        cohort_id="ACTIVE_1975", birth_year=1975, headcount=100,
        gross_salary=100_000.0, accrued_savings=300_000.0, gender="unisex",
    ))
    fund.add_cohort(Cohort(
        cohort_id="RETIRED_1955", birth_year=1955, headcount=50,
        gross_salary=0.0, accrued_savings=0.0, status="retired",
        annual_pension=30_000.0, gender="f",
    ))
    mortality = ek2001_2005(improvement_rate=0.0125, base_year=2003)

    # --- stochastic GBM equity stub (monthly grid, like StochasticALM) ----
    # Only used by the 5c delegation branch; the GC roll uses equity_return.
    M = horizon_years * 12
    rng = np.random.default_rng(7)
    log_ret = rng.normal(0.004, 0.04, size=(M, n_paths))
    levels = 100.0 * np.exp(np.vstack([np.zeros(n_paths), np.cumsum(log_ret, axis=0)]))
    sim_equity = SimpleNamespace(times=np.linspace(0.0, horizon_years, M + 1), rates=levels)

    # --- per-path asset coupons (fixed bond IP) + a noise MD event --------
    asset_cfs = []
    for i in range(n_paths):
        rows = [
            {"time": f"{y}-01-01T00:00:00", "type": "IP",
             "payoff": 3_600_000.0 * (1.0 + 0.01 * i)}
            for y in range(start_year + 1, start_year + horizon_years + 1)
        ]
        rows.append({"time": f"{start_year + horizon_years}-01-01T00:00:00",
                     "type": "MD", "payoff": 180_000_000.0})
        asset_cfs.append(SimpleNamespace(events_df=pd.DataFrame(rows)))

    # --- per-path liability flows + headcount trace -----------------------
    liab_paths = []
    for i in range(n_paths):
        rows = []
        trace = []
        retired_hc = 50.0
        for y in range(start_year, start_year + horizon_years):
            rows.append({"time": f"{y}-06-30T00:00:00", "type": SAV_CONTRIB,
                         "payoff": 2_500_000.0})
            rows.append({"time": f"{y}-12-31T00:00:00", "type": "PENSION_PAYMENT",
                         "payoff": -30_000.0 * retired_hc})
            rows.append({"time": f"{y}-12-31T00:00:00", "type": "RISK_CONTRIB",
                         "payoff": 400_000.0})
            trace.append({"year": y, "birth_year": 1975, "headcount": 100,
                          "status": "active", "gender": "unisex",
                          "agh_per_capita": 300_000.0 + 12_000.0 * (y - start_year),
                          "annual_pension_per_capita": 0.0})
            trace.append({"year": y, "birth_year": 1955, "headcount": retired_hc,
                          "status": "retired", "gender": "f",
                          "agh_per_capita": 0.0,
                          "annual_pension_per_capita": 30_000.0})
            retired_hc = max(retired_hc - (0.8 + 0.2 * i), 0.0)  # path-specific deaths
        liab_paths.append(SimpleNamespace(
            cashflows=SimpleNamespace(events_df=pd.DataFrame(rows)),
            headcount_trace=trace,
        ))

    salm_stub = SimpleNamespace(
        n_paths=n_paths,
        steps_per_year=12,
        base_date=base_date,
        equity_simulation=sim_equity,
        _asset_cfs=asset_cfs,
        _check_ran=lambda: None,
    )

    kwargs = dict(
        fund=fund, mortality=mortality,
        bond_book=180_000_000.0, cash_book=60_000_000.0,
        equity_sleeve_0=60_000_000.0, liab_paths=liab_paths,
    )
    sfra_5c = StochasticFundingRatioAnalysis(salm_stub, **kwargs)
    gc_off = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=None, sanierung=None,
    )

    # ---- Assert 1: regression ------------------------------------------
    dates = [f"{start_year + dy}-01-01" for dy in (0, 1, 3, 5, 7, 9)]
    for d in dates:
        np.testing.assert_array_equal(
            gc_off.funding_ratio_distribution(d),
            sfra_5c.funding_ratio_distribution(d),
            err_msg=f"rebalancing=None must be byte-identical to Stage 5c at {d}",
        )
    print(f"  [OK] Assert 1 — rebalancing=None, sanierung=None == Stage 5c "
          f"(byte-identical on {len(dates)} dates x {n_paths} paths)")

    # ---- active fixed-mix policy must change the roll (and spare t0) ----
    pol_b = RebalancingPolicy(target_weights={BONDS: 0.40, EQUITY: 0.60})
    gc_b = GoingConcernFundingRatioAnalysis(salm_stub, **kwargs, rebalancing=pol_b)
    d_late = f"{start_year + 9}-01-01"
    diff = np.abs(
        gc_b.funding_ratio_distribution(d_late)
        - sfra_5c.funding_ratio_distribution(d_late)
    )
    assert float(diff.max()) > 0.0, (
        "active RebalancingPolicy left the DG distribution unchanged — "
        "the going-concern roll is dead code"
    )
    assert np.array_equal(
        gc_b.funding_ratio_distribution(base_date),
        sfra_5c.funding_ratio_distribution(base_date),
    ), "at t0 (before the first anniversary) the roll must not change V"
    print(f"  [OK] Szenario-B-Mechanik — fixed mix 40/60 shifts DG at {d_late} "
          f"(max |dDG| = {float(diff.max()):.4%}), t0 untouched")

    # ---- Assert 3: |w - target| < 1e-12 after each rebalance -------------
    wp = gc_b.weight_path(0, d_late)
    reb = wp[wp["rebalanced"]]
    assert len(reb) == 9, f"expected 9 yearly rebalances, got {len(reb)}"
    dev = np.max(np.abs(np.r_[reb["w_BONDS"] - reb["target_BONDS"],
                              reb["w_EQUITY"] - reb["target_EQUITY"]]))
    assert dev < 1e-12, f"post-rebalance weight deviation {dev:.2e} >= 1e-12"
    print(f"  [OK] Assert 3 — post-rebalance |w - target| = {dev:.2e} < 1e-12")

    # ---- Assert 5: shift_at_year switches the effective targets ----------
    pol_c = RebalancingPolicy(
        target_weights={BONDS: 0.40, EQUITY: 0.60},
        shift_at_year=5,
        shift_weights={BONDS: 0.60, EQUITY: 0.40},
    )
    gc_c = GoingConcernFundingRatioAnalysis(salm_stub, **kwargs, rebalancing=pol_c)
    wp_c = gc_c.weight_path(0, d_late)
    reb_c = wp_c[wp_c["rebalanced"]].set_index("year_index")
    assert all(abs(reb_c.loc[k, "w_BONDS"] - 0.40) < 1e-12 for k in (1, 2, 3, 4))
    assert all(abs(reb_c.loc[k, "w_BONDS"] - 0.60) < 1e-12 for k in (5, 6, 7, 8, 9))
    print("  [OK] Assert 5 — Gewichtspfad shifts 40/60 -> 60/40 ab Jahr 5")

    # ---- equity_sigma: 0.0 is a byte-identical no-op ----------------------
    gc_sig0 = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_b, equity_sigma=0.0,
    )
    for d in dates:
        np.testing.assert_array_equal(
            gc_sig0.funding_ratio_distribution(d),
            gc_b.funding_ratio_distribution(d),
            err_msg=f"equity_sigma=0.0 must be byte-identical to the drift at {d}",
        )
    print("  [OK] equity_sigma=0.0 — byte-identical to the deterministic drift")

    # ---- equity_sigma > 0: per-path lognormal, SeedSequence-reproducible --
    gc_s1 = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_b, equity_sigma=0.10, equity_seed=123,
    )
    gc_s2 = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_b, equity_sigma=0.10, equity_seed=123,
    )
    gc_s3 = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_b, equity_sigma=0.10, equity_seed=124,
    )
    dist_s1 = gc_s1.funding_ratio_distribution(d_late)
    np.testing.assert_array_equal(
        dist_s1, gc_s2.funding_ratio_distribution(d_late),
        err_msg="same equity_seed must reproduce identical distributions",
    )
    assert not np.array_equal(dist_s1, gc_s3.funding_ratio_distribution(d_late)), (
        "different equity_seed must change the sigma>0 distribution"
    )
    assert not np.array_equal(dist_s1, gc_b.funding_ratio_distribution(d_late)), (
        "equity_sigma=0.10 must differ from the deterministic drift"
    )
    # Path-wise distinct shocks (one spawned generator per path):
    assert np.unique(dist_s1).size > 1, (
        "sigma>0 distribution is degenerate — shocks not per path?"
    )
    print("  [OK] equity_sigma=0.10 — lognormal per path, "
          "SeedSequence-reproduzierbar, seed-sensitiv")

    # ---- bond_yield: None/0.0 no-op; Positivteil-Klemme; >0 wirkt monoton --
    # With pol_b the 40% BONDS target sits far below the 80% start book
    # (240M of 300M), so the bucket never exceeds the initial book value
    # and the positive-part clamp must make bond_yield a no-op:
    gc_clamp = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_b, bond_yield=0.02,
    )
    np.testing.assert_array_equal(
        gc_clamp.funding_ratio_distribution(d_late),
        gc_b.funding_ratio_distribution(d_late),
        err_msg="bucket below initial book: positive-part clamp must "
                "make bond_yield a no-op",
    )
    # With an 80/20 target the bucket stays at/above the start book and
    # grows past it via reinvested net inflows — the yield must bite:
    pol_hi = RebalancingPolicy(target_weights={BONDS: 0.80, EQUITY: 0.20})
    gc_hiN = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_hi,
    )
    gc_hi0 = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_hi, bond_yield=0.0,
    )
    for d in dates:
        np.testing.assert_array_equal(
            gc_hi0.funding_ratio_distribution(d),
            gc_hiN.funding_ratio_distribution(d),
            err_msg=f"bond_yield=0.0 must match bond_yield=None at {d}",
        )
    gc_hi = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_hi, bond_yield=0.02,
    )
    dist_hi = gc_hi.funding_ratio_distribution(d_late)
    dist_h0 = gc_hiN.funding_ratio_distribution(d_late)
    assert not np.array_equal(dist_hi, dist_h0), (
        "bond_yield=0.02 must change the DG distribution once the bucket "
        "exceeds the initial book value"
    )
    assert np.all(dist_hi >= dist_h0 - 1e-15), (
        "bond_yield > 0 with positive-part incremental balance can only "
        "add assets — DG must be pointwise >= the no-yield case"
    )
    print("  [OK] bond_yield — None/0.0 identisch, Positivteil-Klemme greift, "
          f"0.02 wirkt pointwise >= (max dDG = {float(np.max(dist_hi - dist_h0)):.4%})")

    # ---- Baustein 3: SanierungsPolicy --------------------------------------
    # Never fires (trigger 0): byte-identical to sanierung=None, no events.
    gc_san_off = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_b,
        sanierung=SanierungsPolicy(trigger_dg=0.0, exit_dg=0.0),
    )
    for d in dates:
        np.testing.assert_array_equal(
            gc_san_off.funding_ratio_distribution(d),
            gc_b.funding_ratio_distribution(d),
            err_msg=f"never-firing SanierungsPolicy must equal sanierung=None at {d}",
        )
    assert gc_san_off.gc_events(0, d_late).empty, (
        "never-firing SanierungsPolicy must produce no SANIERUNG_SB events"
    )
    print("  [OK] SanierungsPolicy (trigger=0) — nie aktiv, byte-identisch, "
          "gc_events leer")

    # Always fires (trigger 99): SB = sb_factor * SAV_CONTRIB_t, pointwise >=.
    gc_san = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_b,
        sanierung=SanierungsPolicy(trigger_dg=99.0, exit_dg=99.0, sb_factor=0.5),
    )
    dist_san = gc_san.funding_ratio_distribution(d_late)
    dist_ref = gc_b.funding_ratio_distribution(d_late)
    assert not np.array_equal(dist_san, dist_ref), "SB must change the distribution"
    assert np.all(dist_san >= dist_ref - 1e-15), (
        "SB is a pure asset inflow — DG must be pointwise >= without it"
    )
    ev = gc_san.gc_events(1, d_late)
    assert len(ev) == 9, f"expected 9 SB years (k=1..9), got {len(ev)}"
    assert (ev["type"] == "SANIERUNG_SB").all()
    np.testing.assert_allclose(
        ev["payoff"].to_numpy(), 0.5 * 2_500_000.0,
        err_msg="SB_t must equal sb_factor * SAV_CONTRIB_t",
    )
    assert (ev["dg_prev"] < 99.0).all(), "SB only in years with DG_{t-1} < trigger"
    # Liability stream untouched (provenance):
    liab_df_check = liab_paths[1].cashflows.events_df
    assert "SANIERUNG_SB" not in set(liab_df_check["type"]), (
        "SANIERUNG_SB must never appear in the liability stream"
    )
    print("  [OK] SanierungsPolicy (immer aktiv) — SB = 0.5 x SAV_CONTRIB, "
          f"pointwise >= (max dDG = {float(np.max(dist_san - dist_ref)):.4%}), "
          "Log separat")

    # max_years caps the number of SB-paying years per path.
    gc_san_cap = GoingConcernFundingRatioAnalysis(
        salm_stub, **kwargs, rebalancing=pol_b,
        sanierung=SanierungsPolicy(trigger_dg=99.0, exit_dg=99.0,
                                   sb_factor=0.5, max_years=3),
    )
    assert len(gc_san_cap.gc_events(0, d_late)) == 3, "max_years=3 must cap SB years"
    print("  [OK] SanierungsPolicy max_years=3 — genau 3 SB-Jahre")

    # ---- guards -----------------------------------------------------------
    try:
        GoingConcernFundingRatioAnalysis(salm_stub, **kwargs, sanierung=object())
    except TypeError:
        print("  [OK] sanierung mit falschem Typ raises TypeError")
    else:
        raise AssertionError("non-SanierungsPolicy sanierung must raise TypeError")

    try:
        GoingConcernFundingRatioAnalysis(
            salm_stub, **kwargs, sanierung=SanierungsPolicy(),
        )
    except ValueError:
        print("  [OK] sanierung ohne RebalancingPolicy raises ValueError")
    else:
        raise AssertionError("sanierung without rebalancing must raise")

    try:
        SanierungsPolicy(trigger_dg=1.0, exit_dg=0.9)
    except ValueError:
        print("  [OK] exit_dg < trigger_dg raises ValueError")
    else:
        raise AssertionError("exit_dg < trigger_dg must raise")

    try:
        GoingConcernFundingRatioAnalysis(salm_stub, **kwargs, equity_sigma=0.10)
    except ValueError:
        print("  [OK] equity_sigma > 0 ohne RebalancingPolicy raises ValueError")
    else:
        raise AssertionError("equity_sigma>0 without rebalancing must raise")

    try:
        GoingConcernFundingRatioAnalysis(salm_stub, **kwargs, bond_yield=0.02)
    except ValueError:
        print("  [OK] bond_yield ohne RebalancingPolicy raises ValueError")
    else:
        raise AssertionError("bond_yield without rebalancing must raise")

    try:
        RebalancingPolicy(target_weights={BONDS: 0.50, EQUITY: 0.60})
    except ValueError:
        print("  [OK] target_weights sum != 1 raises ValueError")
    else:
        raise AssertionError("non-normalized target_weights must raise")


if __name__ == "__main__":
    print("Stage 5d Baustein 1 — Akzeptanz-Asserts (offline) vs Stage 5c")
    print("-" * 64)
    _regression_check_against_5c()
    print("All Baustein-1 asserts passed.")
