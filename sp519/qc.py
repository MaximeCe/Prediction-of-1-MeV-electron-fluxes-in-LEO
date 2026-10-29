"""Quality report of the modelling table, and the event threshold of Step 2.

Everything returned here is plain Python (floats, ints, strings) so that it can be dumped
into the JSON report shipped next to the parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def event_threshold(values, q: float) -> dict:
    """``q``-th percentile of ``values``: above it, an interval is an "event"."""
    s = pd.Series(values).dropna()
    if s.empty:
        raise ValueError("cannot compute an event threshold on an empty series")
    thr = float(np.percentile(s.to_numpy(dtype=float), q))
    return {
        "q": float(q),
        "threshold": thr,
        "log10": float(np.log10(thr)) if thr > 0 else float("-inf"),
        "n": int(len(s)),
        "n_events": int((s > thr).sum()),
        "fraction_events": float((s > thr).mean()),
    }


def coverage_report(long: pd.DataFrame) -> pd.DataFrame:
    """Per L bin: how many intervals, and how well they are sampled."""
    g = long.groupby(level="Lbin", observed=True)
    return pd.DataFrame(
        {
            "n_intervals": g.size().astype(int),
            "median_coverage": g["coverage"].median(),
            "min_coverage": g["coverage"].min(),
            "median_samples": g["n_samples"].median(),
            "median_flux": g["p6_flux"].median(),
        }
    )


def summary(long: pd.DataFrame, cfg: Config) -> dict:
    """The headline numbers of Step 1, ready for the report and the notebook."""
    flux = long["p6_flux"]
    thr = event_threshold(flux, cfg.event_percentile)
    l_bins = sorted(float(x) for x in long.index.get_level_values("Lbin").unique())
    return {
        "freq": cfg.freq,
        "l_range": [float(cfg.l_range[0]), float(cfg.l_range[1])],
        "l_bins": l_bins,
        "n_l_bins": int(len(l_bins)),
        "n_rows": int(len(long)),
        "n_intervals": int(long.index.get_level_values("time_bin").nunique()),
        "start": str(long.index.get_level_values("time_bin").min()),
        "end": str(long.index.get_level_values("time_bin").max()),
        "median_coverage": float(long["coverage"].median()),
        "min_coverage": float(long["coverage"].min()),
        "median_flux": float(flux.median()),
        "event_percentile": float(cfg.event_percentile),
        "event_threshold": thr["threshold"],
        "event_threshold_log10": thr["log10"],
        "n_events": thr["n_events"],
        "fraction_events": thr["fraction_events"],
    }


def format_summary(summary: dict) -> str:
    """Human-readable rendering of :func:`summary` (used by the CLI and the notebook)."""
    lines = [
        f"period        : {summary['start']} -> {summary['end']}",
        f"resolution    : {summary['freq']}",
        f"L range       : {summary['l_range'][0]:g} - {summary['l_range'][1]:g} "
        f"({summary['n_l_bins']} bins)",
        f"table         : {summary['n_rows']} rows = {summary['n_intervals']} intervals "
        f"x {summary['n_l_bins']} L bins",
        f"coverage      : median {summary['median_coverage']:.3f}, "
        f"min {summary['min_coverage']:.3f}",
        f"median flux   : {summary['median_flux']:.1f} #/cm2/s/sr",
        f"event         : p{summary['event_percentile']:g} = "
        f"{summary['event_threshold']:.1f} #/cm2/s/sr "
        f"(log10 {summary['event_threshold_log10']:.2f}), "
        f"{summary['n_events']} / {summary['n_rows']} rows "
        f"({100 * summary['fraction_events']:.2f} %)",
    ]
    return "\n".join(lines)
