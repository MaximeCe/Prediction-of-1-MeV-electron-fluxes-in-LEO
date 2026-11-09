"""12 h binning and the synchronization of the two sources into one table.

Output shape (the "long" table) is one row per ``(Lbin, 12 h interval)``:

===================  =========================================================
``p6_flux``          mean of the hourly P6 flux over the interval [#/cm2/s/sr]
``log10_p6``         log10 of ``p6_flux``, with a half-count floor for zeros
``n_samples``        raw 2 s measurements behind the interval
``n_hours``          hours of the interval that were really observed
``coverage``         ``n_hours`` / hours expected -> how much the interval is trusted
``contaminated_frac``share of the interval flagged as proton-contaminated
OMNI columns         mean of the solar wind / indices over the same interval
===================  =========================================================

A "wide" view (one row per interval, one column per L bin) is provided as well, mostly for
a quick look and for the figures; the long one is what Step 2 consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io
from .config import Config

FLUX = "p6_flux"
LOG_FLUX = "log10_p6"
OMNI_JOINED = io.OMNI_COLUMNS  # columns copied from OMNI onto every L bin of an interval
LONG_META = (FLUX, LOG_FLUX, "n_samples", "n_hours", "coverage", "contaminated_frac")


def to_long_table(noaa: pd.DataFrame, omni: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Bin the cleaned NOAA table over ``cfg.freq`` intervals and attach the OMNI means."""
    expected = cfg.expected_hours_per_bin

    frame = noaa.reset_index()
    grouped = frame.groupby(
        ["Lbin", pd.Grouper(key="time", freq=cfg.freq, origin="epoch")],
        observed=True,
    )
    out = grouped.agg(
        p6_flux=("flux_hourly", "mean"),
        n_samples=("n", "sum"),
        n_hours=("flux_hourly", "count"),
        contaminated_frac=("contaminated", "mean"),
    )
    out.index = out.index.set_names(["Lbin", "time_bin"])

    out["coverage"] = out["n_hours"] / expected
    if cfg.min_coverage > 0:
        out = out[out["coverage"] >= cfg.min_coverage]
    if out.empty:
        raise ValueError(
            f"no interval survives min_coverage={cfg.min_coverage} "
            f"(freq={cfg.freq!r}, l_range={cfg.l_range})"
        )

    out[LOG_FLUX] = np.log10(np.maximum(out[FLUX].to_numpy(), cfg.flux_floor))
    out = out.reset_index()

    omni_binned = omni.resample(cfg.freq, origin="epoch").mean()
    omni_binned.index = omni_binned.index.set_names("time_bin")
    out = out.merge(omni_binned.reset_index(), on="time_bin", how="left")

    columns = ["Lbin", "time_bin", FLUX, LOG_FLUX, "n_samples", "n_hours", "coverage",
               "contaminated_frac", *[c for c in OMNI_JOINED if c in out.columns]]
    return out[columns].set_index(["Lbin", "time_bin"]).sort_index()


def to_wide_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per 12 h interval, one ``p6_L<l>`` column per L bin, plus the OMNI means."""
    wide = long[FLUX].unstack("Lbin")
    wide.columns = [f"p6_L{c:g}" for c in wide.columns]

    extra = [c for c in long.columns if c in OMNI_JOINED]
    if extra:
        first_lbin = long.index.get_level_values("Lbin").unique()[0]
        wide = wide.join(long.xs(first_lbin, level="Lbin")[extra], how="left")
    wide = wide.join(long["coverage"].unstack("Lbin").mean(axis=1).rename("coverage"))
    wide.index.name = "time_bin"
    return wide.sort_index()
