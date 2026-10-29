"""Figures of Step 1 ("visualize the time series", "study the relationships").

Every function takes the long table produced by ``sp519.aggregate.to_long_table`` and returns
a matplotlib Figure, so the notebook can display it and the CLI can save it.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .aggregate import FLUX, LOG_FLUX, OMNI_JOINED  # noqa: E402
from .config import Config  # noqa: E402


def _l_bins(long: pd.DataFrame) -> list[float]:
    return sorted(float(x) for x in long.index.get_level_values("Lbin").unique())


def _matrix(long: pd.DataFrame, column: str) -> pd.DataFrame:
    m = long[column].unstack("Lbin")
    m.columns = [float(c) for c in m.columns]
    return m.sort_index(axis=1)


# --------------------------------------------------------------------------------------
def plot_timeseries(long: pd.DataFrame) -> plt.Figure:
    """log10 P6 flux in every L bin vs time, with the coverage of each interval below."""
    flux = _matrix(long, LOG_FLUX).sort_index()
    coverage = long["coverage"].groupby(level="time_bin").median().sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for l_bin in flux.columns:
        axes[0].plot(flux.index, flux[l_bin], lw=0.7, label=f"L={l_bin:g}")
    axes[0].set_ylabel("log10 P6 flux\n[#/cm²/s/sr]")
    axes[0].set_title("1 MeV electron flux (NOAA-15 P6) per L bin")
    axes[0].legend(ncol=2, fontsize=7)

    axes[1].plot(coverage.index, coverage.to_numpy(), lw=0.7, color="C7")
    axes[1].set_ylabel("coverage")
    axes[1].set_xlabel("time (UTC)")
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def plot_flux_vs_l(long: pd.DataFrame, max_points: int = 30_000) -> plt.Figure:
    """Where the belts are: flux against L, to justify the selected L range."""
    d = long.reset_index()
    if len(d) > max_points:
        d = d.iloc[:: max(1, len(d) // max_points)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(d["Lbin"], d[LOG_FLUX], s=2, alpha=0.15, color="C0")
    prof = long[LOG_FLUX].groupby(level="Lbin").median()
    ax.plot(prof.index.astype(float), prof.to_numpy(), "r-o", ms=4, lw=1.5, label="median per L bin")
    ax.set_xlabel("L (McIlwain)")
    ax.set_ylabel("log10 P6 flux")
    ax.legend(fontsize=8)
    ax.set_title("Radiation belt profile: the outer belt peaks near L = 4.5")
    fig.tight_layout()
    return fig


def plot_distribution(long: pd.DataFrame, threshold: float | None = None) -> plt.Figure:
    """Distribution of the 12 h flux with the 95th-percentile event threshold."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(long[LOG_FLUX].dropna(), bins=60, color="C0", alpha=0.85)
    if threshold is not None:
        ax.axvline(np.log10(threshold), color="r", ls="--",
                   label=f"p95 = {threshold:.0f} #/cm²/s/sr")
        ax.legend()
    ax.set_xlabel("log10 P6 flux")
    ax.set_ylabel("intervals")
    ax.set_title("12 h flux distribution")
    fig.tight_layout()
    return fig


def plot_coverage(long: pd.DataFrame) -> plt.Figure:
    """Share of each 12 h interval that was really observed, per L bin."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for l_bin in _l_bins(long):
        sub = long.xs(l_bin, level="Lbin")["coverage"].sort_index()
        ax.plot(sub.index, sub.to_numpy(), lw=0.6, label=f"L={l_bin:g}")
    ax.set_ylabel("coverage (observed hours / 12)")
    ax.set_xlabel("time (UTC)")
    ax.legend(ncol=2, fontsize=7)
    ax.set_title("Sampling coverage of the 12 h intervals")
    fig.tight_layout()
    return fig


def plot_correlation(long: pd.DataFrame, method: str = "spearman") -> plt.Figure:
    """Correlation heatmap between the flux, L and the solar parameters (Spearman)."""
    cols = [LOG_FLUX, "Lbin", *[c for c in OMNI_JOINED if c in long.columns]]
    d = long.reset_index()[cols].copy()
    c = d.corr(method=method)

    fig, ax = plt.subplots(figsize=(1.4 + 0.85 * len(c), 1.0 + 0.75 * len(c)))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.to_numpy()[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=f"{method} correlation")
    fig.tight_layout()
    return fig


def plot_lag_correlation(long: pd.DataFrame, max_lag: int = 8, method: str = "spearman") -> plt.Figure:
    """Flux(t) against each solar parameter at t - lag: how far ahead a driver acts."""
    tmp = long.reset_index().sort_values(["Lbin", "time_bin"])
    l_bins = _l_bins(long)
    median_l = l_bins[len(l_bins) // 2]
    d = tmp[tmp["Lbin"] == median_l].set_index("time_bin")
    flux = d[FLUX]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for c in [c for c in OMNI_JOINED if c in d.columns]:
        ax.plot(range(max_lag + 1), [flux.corr(d[c].shift(k), method=method) for k in range(max_lag + 1)],
                "o-", ms=3, label=c)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("solar parameter delayed by (x 12 h)")
    ax.set_ylabel(f"{method} correlation with flux (L={median_l:g})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def save_all(long: pd.DataFrame, cfg: Config, outdir: str | None = None) -> dict:
    """Build every figure and write the PNGs; returns ``{name: Figure}``."""
    from . import qc

    threshold = qc.event_threshold(long[FLUX], cfg.event_percentile)["threshold"]
    figs = {
        "timeseries": plot_timeseries(long),
        "flux_vs_L": plot_flux_vs_l(long),
        "distribution": plot_distribution(long, threshold),
        "coverage": plot_coverage(long),
        "correlation": plot_correlation(long),
        "lag_correlation": plot_lag_correlation(long),
    }
    outdir = outdir or cfg.figures_dir
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=130, bbox_inches="tight")
    return figs
