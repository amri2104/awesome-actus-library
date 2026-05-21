"""Stage 1 liability plotting helpers.

All functions accept *events_df* (the DataFrame from CashFlowStream.events_df)
and an optional *output_path* (Path or str). When output_path is given the
figure is saved and closed; otherwise it is returned open for interactive use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..events import (
    ADMIN_COST,
    INTEREST_CREDIT,
    PENSION_PAYMENT,
    RETIREMENT_CONV,
    RISK_CONTRIB,
    SAV_CONTRIB,
)

_COLORS = {
    SAV_CONTRIB: "#4878CF",
    RISK_CONTRIB: "#92B4E3",
    ADMIN_COST: "#F28E2B",
    PENSION_PAYMENT: "#C0392B",
}

_HEADCOUNT_COLORS = {"active": "#4878CF", "retired": "#C0392B"}

_CHF_FMT = mticker.FuncFormatter(lambda x, _: f"CHF {x/1e6:.1f}M" if abs(x) >= 1e6 else f"CHF {x/1e3:.0f}k")


def _year_col(events_df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(events_df["time"]).dt.year


def _save_or_return(fig, output_path):
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_annual_cashflows_by_type(
    events_df: pd.DataFrame,
    output_path=None,
    title: str = "Annual Cashflows by Event Type",
) -> Optional[plt.Figure]:
    """Stacked bar: inflows above zero, outflows below zero, by event type."""
    df = events_df.copy()
    df["year"] = _year_col(df)

    cf_types = [SAV_CONTRIB, RISK_CONTRIB, ADMIN_COST, PENSION_PAYMENT]
    df = df[df["type"].isin(cf_types)]
    pivot = df.groupby(["year", "type"])["payoff"].sum().unstack(fill_value=0.0)

    years = pivot.index.values
    x = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom_pos = np.zeros(len(years))
    bottom_neg = np.zeros(len(years))

    for etype in [SAV_CONTRIB, RISK_CONTRIB]:
        if etype not in pivot.columns:
            continue
        vals = pivot[etype].values
        ax.bar(x, vals, bottom=bottom_pos, label=etype, color=_COLORS[etype], width=0.7)
        bottom_pos += vals

    for etype in [PENSION_PAYMENT, ADMIN_COST]:
        if etype not in pivot.columns:
            continue
        vals = pivot[etype].values
        ax.bar(x, vals, bottom=bottom_neg, label=etype, color=_COLORS[etype], width=0.7)
        bottom_neg += vals

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x[::5])
    ax.set_xticklabels(years[::5], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(_CHF_FMT)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cashflow (CHF)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_net_liability_cashflow(
    events_df: pd.DataFrame,
    output_path=None,
    title: str = "Annual Net Liability Cashflow",
) -> Optional[plt.Figure]:
    """Bar chart of net fund cashflow per year (positive = net inflow)."""
    df = events_df.copy()
    df["year"] = _year_col(df)
    net = df.groupby("year")["payoff"].sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#4878CF" if v >= 0 else "#C0392B" for v in net.values]
    ax.bar(net.index, net.values, color=colors, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(_CHF_FMT)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Net Cashflow (CHF)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_cumulative_net_cashflow(
    events_df: pd.DataFrame,
    output_path=None,
    title: str = "Cumulative Net Liability Cashflow",
) -> Optional[plt.Figure]:
    """Line chart of cumulative net cashflow over simulation horizon."""
    df = events_df.copy()
    df["year"] = _year_col(df)
    net = df.groupby("year")["payoff"].sum().sort_index()
    cumnet = net.cumsum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cumnet.index, cumnet.values, marker="o", markersize=3, color="#4878CF", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(cumnet.index, cumnet.values, 0,
                    where=(cumnet.values >= 0), alpha=0.15, color="#4878CF")
    ax.fill_between(cumnet.index, cumnet.values, 0,
                    where=(cumnet.values < 0), alpha=0.15, color="#C0392B")
    ax.yaxis.set_major_formatter(_CHF_FMT)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Cashflow (CHF)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_agh_path(
    events_df: pd.DataFrame,
    output_path=None,
    title: str = "AGH (Altersgutscha ben) per Capita by Cohort",
) -> Optional[plt.Figure]:
    """Line per active cohort showing AGH accumulation; dot at retirement conversion.

    Requires *agh_per_capita* field in INTEREST_CREDIT events (emitted by
    ClosedFundSimulator). Cohorts already retired at simulation start (no
    INTEREST_CREDIT rows) are skipped.
    """
    ic = events_df[events_df["type"] == INTEREST_CREDIT].copy()
    rc = events_df[events_df["type"] == RETIREMENT_CONV].copy()

    if ic.empty and rc.empty:
        return None

    ic["year"] = _year_col(ic)
    rc["year"] = _year_col(rc)

    cohorts_with_active = ic["contractId"].unique() if not ic.empty else []

    cmap = plt.cm.get_cmap("tab10", max(len(cohorts_with_active), 1))

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, cid in enumerate(cohorts_with_active):
        sub_ic = ic[ic["contractId"] == cid].sort_values("year")
        if "agh_per_capita" not in sub_ic.columns or sub_ic["agh_per_capita"].isna().all():
            continue
        color = cmap(idx)
        ax.plot(sub_ic["year"], sub_ic["agh_per_capita"],
                marker=".", markersize=4, linewidth=1.5, color=color, label=cid)

        sub_rc = rc[rc["contractId"] == cid]
        if not sub_rc.empty:
            row = sub_rc.iloc[0]
            ax.scatter([row["year"]], [row["agh_per_capita_at_conversion"]],
                       marker="*", s=150, color=color, zorder=5)

    if len(cohorts_with_active) == 0:
        ax.text(0.5, 0.5, "No active cohorts — all cohorts already retired at t=0",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")

    ax.yaxis.set_major_formatter(_CHF_FMT)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("AGH per Capita (CHF)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_all(
    events_df: pd.DataFrame,
    output_dir,
    prefix: str = "stage1",
) -> None:
    """Save all Stage 1 plots to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_annual_cashflows_by_type(events_df, output_path=out / f"{prefix}_cashflows_by_type.png")
    plot_net_liability_cashflow(events_df, output_path=out / f"{prefix}_net_cashflow.png")
    plot_cumulative_net_cashflow(events_df, output_path=out / f"{prefix}_cumulative_cashflow.png")
    plot_agh_path(events_df, output_path=out / f"{prefix}_agh_path.png")
    print(f"  [plots] saved 4 figures to {out}/")


def plot_headcount(
    headcount_df: pd.DataFrame,
    output_path=None,
    title: str = "Active vs Retired Headcount by Year",
) -> Optional[plt.Figure]:
    """Stacked area chart of active and retired headcount per simulation year.

    Args:
        headcount_df: pd.DataFrame with columns [year, active, retired],
                      typically pd.DataFrame(simulator.headcount_log).
    """
    df = headcount_df.sort_values("year")
    years = df["year"].values
    active = df["active"].values
    retired = df["retired"].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(
        years,
        active,
        retired,
        labels=["Active", "Retired"],
        colors=[_HEADCOUNT_COLORS["active"], _HEADCOUNT_COLORS["retired"]],
        alpha=0.80,
    )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Headcount")
    ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_cumulative_comparison(
    closed_events_df: pd.DataFrame,
    open_events_df: pd.DataFrame,
    output_path=None,
    title: str = "Cumulative Net Cashflow: Closed vs Open Fund",
) -> Optional[plt.Figure]:
    """Two-line comparison of cumulative net cashflow: closed fund vs open fund."""

    def _annual_net(df: pd.DataFrame) -> pd.Series:
        d = df.copy()
        d["year"] = pd.to_datetime(d["time"]).dt.year
        return d.groupby("year")["payoff"].sum().sort_index()

    closed_ann = _annual_net(closed_events_df)
    open_ann = _annual_net(open_events_df)

    all_years = closed_ann.index.union(open_ann.index)
    closed_ann = closed_ann.reindex(all_years, fill_value=0.0)
    open_ann = open_ann.reindex(all_years, fill_value=0.0)

    closed_cum = closed_ann.cumsum()
    open_cum = open_ann.cumsum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(closed_cum.index, closed_cum.values,
            marker="o", markersize=3, linewidth=1.5, color="#4878CF", label="Closed Fund")
    ax.plot(open_cum.index, open_cum.values,
            marker="s", markersize=3, linewidth=1.5, color="#2CA02C", label="Open Fund")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(open_cum.index, open_cum.values, closed_cum.values,
                    alpha=0.10, color="#2CA02C", label="Open − Closed")
    ax.yaxis.set_major_formatter(_CHF_FMT)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Cashflow (CHF)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_all_stage2(
    events_df: pd.DataFrame,
    headcount_df: pd.DataFrame,
    closed_events_df: pd.DataFrame,
    output_dir,
    prefix: str = "stage2",
) -> None:
    """Save all Stage 1 plots (for open fund) plus 2 Stage 2 plots to *output_dir*.

    Args:
        events_df:        events_df from OpenFundSimulator (used for stage1-style plots).
        headcount_df:     pd.DataFrame(simulator.headcount_log).
        closed_events_df: events_df from ClosedFundSimulator (for comparison plot).
        output_dir:       Directory to write PNG files into.
        prefix:           Filename prefix for all output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_annual_cashflows_by_type(events_df, output_path=out / f"{prefix}_cashflows_by_type.png")
    plot_net_liability_cashflow(events_df, output_path=out / f"{prefix}_net_cashflow.png")
    plot_cumulative_net_cashflow(events_df, output_path=out / f"{prefix}_cumulative_cashflow.png")
    plot_agh_path(events_df, output_path=out / f"{prefix}_agh_path.png")
    plot_headcount(headcount_df, output_path=out / f"{prefix}_headcount.png")
    plot_cumulative_comparison(
        closed_events_df,
        events_df,
        output_path=out / f"{prefix}_cumulative_comparison.png",
    )
    print(f"  [plots] saved 6 figures to {out}/")
