"""
visualizer.py
=============
Statistics and figures of the 12-hour table (flux, L, solar parameters).

Data: ``load_binned()`` = ``biner.bin_mean(loader.load_cleaned_data())``. Every function takes that
DataFrame (or the 2 s cleaned data for ``plot_timeseries``). Solar parameters are all the columns
that are not flux / L / ``n_samples`` (they are attached by ``cleaner.clean_dataset``).

Statistics                                Figures (return a matplotlib Figure)
----------------------------------------  ---------------------------------------------
describe            descriptive stats      plot_timeseries         flux, L, solar vs time
correlation_matrix  flux vs solar          plot_flux_distribution  histogram + threshold
lag_correlation     flux(t) vs solar(t-k)  plot_correlation        correlation heatmap
event_threshold     95th percentile        plot_flux_vs_solar      scatter flux vs each parameter
outlier_summary     IQR outliers           plot_lag_correlation    lag correlation curves
                                           plot_all                everything, saved as PNG
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleaner import L_COL, OUTLIER_K, P6_COUNT_STEP, RAW_COLUMNS, TARGET

FLUX_FLOOR = P6_COUNT_STEP / 2  # a "0 count" value is drawn at half a count on log axes


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_binned(freq: str = "12h") -> pd.DataFrame:
    """12 h table built from the cleaned dataset written by cleaner.py."""
    from biner import bin_mean
    from loader import load_cleaned_data

    return bin_mean(load_cleaned_data(), freq)


def solar_columns(df: pd.DataFrame) -> list[str]:
    """Solar parameters = columns that are neither flux / L / IFC flag nor ``n_samples``."""
    return [c for c in df.columns if c not in (*RAW_COLUMNS, "n_samples")]


def _log(x):
    return np.log10(np.maximum(x, FLUX_FLOOR))


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def describe(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Descriptive statistics of the flux (linear and log10) and of the solar parameters."""
    cols = list(cols or [TARGET, *solar_columns(df)])
    x = df[cols].copy()
    x["log10_" + TARGET] = _log(x[TARGET])
    out = x.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    out["skew"], out["kurtosis"] = x.skew(), x.kurt()
    return out


def event_threshold(df: pd.DataFrame, q: float = 95.0, col: str = TARGET) -> dict:
    """q-th percentile of the flux: above it, an interval is an "event"."""
    s = df[col].dropna()
    thr = float(np.percentile(s, q))
    return {"q": q, "threshold": thr, "log10": float(_log(thr)), "n": len(s),
            "n_events": int((s > thr).sum()), "fraction_events": float((s > thr).mean())}


def outlier_summary(df: pd.DataFrame, k: float = OUTLIER_K, col: str = TARGET, top: int = 10):
    """IQR rule on log10 flux (same rule as the cleaner). Returns ``(summary, extreme_values)``."""
    x = _log(df[col].dropna())
    q1, q3 = x.quantile([0.25, 0.75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    out = x[(x < lo) | (x > hi)]
    summary = {"k": k, "lower_log10": float(lo), "upper_log10": float(hi), "n": len(x),
               "n_low": int((x < lo).sum()), "n_high": int((x > hi).sum())}
    return summary, out.sort_values(key=abs, ascending=False)[:top].rename("log10_flux").to_frame()


def correlation_matrix(df: pd.DataFrame, cols=None, method: str = "spearman") -> pd.DataFrame:
    """Correlation between the flux (log10), L and the solar parameters (Spearman: heavy tails)."""
    cols = list(cols or [TARGET, L_COL, *solar_columns(df)])
    x = df[cols].copy()
    x[TARGET] = _log(x[TARGET])
    return x.corr(method=method)


def lag_correlation(df: pd.DataFrame, max_lag: int = 8, method: str = "spearman") -> pd.DataFrame:
    """Correlation between flux(t) and each solar parameter at t - lag (lag in hours).

    The flux reacts to the solar wind with a delay, which is what makes it predictable.
    """
    step_h = (df.index[1] - df.index[0]) / pd.Timedelta(hours=1)
    flux = _log(df[TARGET])
    res = {c: [flux.corr(df[c].shift(k), method=method) for k in range(max_lag + 1)] for c in solar_columns(df)}
    return pd.DataFrame(res, index=pd.Index([k * step_h for k in range(max_lag + 1)], name="lag_h"))


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_timeseries(df: pd.DataFrame, threshold: float | None = None, max_points: int = 100_000):
    """Flux, L-shell and each solar parameter vs time. ``threshold`` (linear flux) draws the event line."""
    cols = solar_columns(df)
    d = df.iloc[:: max(1, len(df) // max_points)]
    fig, axes = plt.subplots(2 + len(cols), 1, figsize=(11, 2.5 * (2 + len(cols))), sharex=True, squeeze=False)
    axes = axes[:, 0]
    axes[0].plot(d.index, _log(d[TARGET]), ".-", ms=3, lw=0.6)
    if threshold is not None:
        axes[0].axhline(_log(threshold), color="r", ls="--", lw=1, label="event threshold")
        axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("log10 P6 flux")
    axes[1].plot(d.index, d[L_COL], ".-", ms=3, lw=0.6, color="C2")
    axes[1].set_ylabel("L (IGRF)")
    for ax, c in zip(axes[2:], cols):
        ax.plot(d.index, d[c], ".-", ms=3, lw=0.6, color="C1")
        ax.set_ylabel(c)
    axes[-1].set_xlabel("time (UTC)")
    fig.tight_layout()
    return fig


def plot_flux_distribution(df: pd.DataFrame, q: float = 95.0):
    """Histogram of log10 flux with the q-th percentile threshold."""
    thr = event_threshold(df, q)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(_log(df[TARGET].dropna()), bins=50, color="C0", alpha=0.8)
    ax.axvline(thr["log10"], color="r", ls="--", label=f"p{q:g} = {thr['threshold']:.0f} #/cm²/s/sr")
    ax.set_xlabel("log10 P6 flux"); ax.set_ylabel("count"); ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame, cols=None, method: str = "spearman"):
    """Annotated correlation heatmap."""
    c = correlation_matrix(df, cols, method)
    fig, ax = plt.subplots(figsize=(1.5 + 0.9 * len(c), 1.0 + 0.8 * len(c)))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"{method} correlation")
    fig.tight_layout()
    return fig


def plot_flux_vs_solar(df: pd.DataFrame):
    """log10 flux against each solar parameter."""
    cols = solar_columns(df)
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.6), squeeze=False)
    for ax, c in zip(axes[0], cols):
        ax.plot(df[c], _log(df[TARGET]), ".", ms=4, alpha=0.6)
        ax.set_xlabel(c); ax.set_ylabel("log10 P6 flux")
    fig.tight_layout()
    return fig


def plot_lag_correlation(df: pd.DataFrame, max_lag: int = 8):
    """Correlation of the flux with each solar parameter delayed by 0 ... max_lag intervals."""
    lc = lag_correlation(df, max_lag)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in lc.columns:
        ax.plot(lc.index, lc[c], "o-", label=c)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("solar parameter delayed by (h)"); ax.set_ylabel("Spearman correlation with flux")
    ax.legend(); fig.tight_layout()
    return fig


def plot_all(df: pd.DataFrame, outdir: str | None = "figures", q: float = 95.0) -> dict:
    """Every figure (the solar ones only if solar columns are present); PNG files in ``outdir``."""
    thr = event_threshold(df, q)["threshold"]
    figs = {"timeseries": plot_timeseries(df, thr), "distribution": plot_flux_distribution(df, q)}
    if solar_columns(df):
        figs.update(correlation=plot_correlation(df), flux_vs_solar=plot_flux_vs_solar(df),
                    lag_correlation=plot_lag_correlation(df))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=130)
    return figs


# --------------------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 140, "display.max_columns", 20, "display.float_format", "{:.4g}".format)

    df = load_binned()  # cleaned data (cleaner.py) averaged over 12 h (biner.py)
    print(f"{len(df)} intervals of 12 h, {df.index[0]} -> {df.index[-1]}")
    solar = solar_columns(df)
    print("solar parameters:", solar or "none (cleaner.py was run without solar data)")

    print("\n-- descriptive statistics --")
    print(describe(df))

    thr = event_threshold(df, 95)
    print(f"\n-- event threshold (95th percentile): {thr['threshold']:.1f} #/cm2/s/sr "
          f"(log10 = {thr['log10']:.2f}); {thr['n_events']} / {thr['n']} intervals above")

    summ, extreme = outlier_summary(df)
    print(f"\n-- outliers left after cleaning (IQR, k={summ['k']}): {summ['n_high']} high / {summ['n_low']} low")
    print(extreme.head())

    if solar:
        print("\n-- Spearman correlation --")
        print(correlation_matrix(df).round(2))
        print("\n-- flux(t) vs solar(t - lag) --")
        print(lag_correlation(df).round(2))

    plot_all(df, "figures")
    print("\nfigures saved in ./figures")
    plt.show()