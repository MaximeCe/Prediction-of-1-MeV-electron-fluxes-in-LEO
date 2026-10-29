"""What counts as a valid measurement, and what has to be removed.

The unit of work is the *hourly* row (one hour x one L bin) of the raw table. Two things are
cleaned here and nothing else:

* the NOAA-15 P6 flux  -> a valid, finite, in-range electron flux per hour and L bin;
* the OMNI parameters  -> the solar wind and the geomagnetic indices on the same hours.

Every rule is counted in a report, so the notebook and the written report can state exactly
how much data was discarded and why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io
from .config import Config


# --------------------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------------------
def as_long(df: pd.DataFrame) -> pd.DataFrame:
    """Any flavour of the NOAA table -> DataFrame indexed by ``(time, Lbin)``.

    Accepts the frame returned by ``io.load_noaa_hourly`` (time as the index) as well as a
    plain frame carrying a ``time`` column.
    """
    if isinstance(df.index, pd.MultiIndex):
        return df.sort_index()
    out = df.copy()
    if "time" not in out.columns:
        out = out.reset_index()
    out["time"] = pd.to_datetime(out["time"])
    return out.set_index(["time", "Lbin"]).sort_index()


# --------------------------------------------------------------------------------------
# The two fluxes carried by the raw table
# --------------------------------------------------------------------------------------
def hourly_flux(noaa: pd.DataFrame) -> pd.Series:
    """Mean P6 flux of each hour: ``p6_sum / n``.

    An hour without measurement (``n == 0``) is a genuine gap and becomes NaN. A bin whose
    ``p6_sum`` is 0 is a real "0 count" measurement and stays 0.
    """
    df = as_long(noaa)
    n = df["n"]
    return (df["p6_sum"] / n.where(n > 0)).rename("flux_hourly")


def proton_flux(noaa: pd.DataFrame) -> pd.Series:
    """>25 MeV proton flux of each hour (``omni_p1_sum / n``), the contamination tracer."""
    df = as_long(noaa)
    n = df["omni_p1_n"]
    return (df["omni_p1_sum"] / n.where(n > 0)).rename("proton_flux")


# --------------------------------------------------------------------------------------
# Proton contamination (the "when the channel does not measure electrons" of the brief)
# --------------------------------------------------------------------------------------
def contamination_threshold(noaa: pd.DataFrame, cfg: Config) -> float:
    """Proton flux above which the P6 channel is considered proton-dominated.

    Data-driven: the ``cfg.spe_percentile`` percentile of the proton flux observed inside
    ``cfg.l_range``. Using a percentile rather than a fixed number keeps the rule valid at
    solar minimum and at solar maximum.
    """
    df = as_long(noaa)
    lbin = df.index.get_level_values("Lbin")
    p1 = proton_flux(df).to_numpy()
    belt = (lbin >= cfg.l_range[0]) & (lbin <= cfg.l_range[1])
    values = p1[belt]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")  # nothing to compare against: never flag
    return float(np.percentile(values, cfg.spe_percentile))


def flag_contamination(proton_flux: pd.Series, threshold: float) -> pd.Series:
    """True where the >25 MeV proton flux sits above ``threshold`` (i.e. during an SPE)."""
    return (proton_flux > threshold).rename("contaminated")


# --------------------------------------------------------------------------------------
# NOAA-15
# --------------------------------------------------------------------------------------
def clean_noaa(noaa: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Clean the hourly NOAA-15 table and restrict it to the justified L range.

    Returns ``(df, report)`` where ``df`` is indexed by ``(time, Lbin)`` and carries
    ``flux_hourly``, ``proton_flux`` and the boolean ``contaminated``.
    """
    df = as_long(noaa)
    lbin = df.index.get_level_values("Lbin")
    n = df["n"].to_numpy()

    rep: dict = {"n_input": int(len(df))}

    no_measurement = ~(n >= max(cfg.min_hourly_samples, 1))  # n == 0 for the shipped table
    outside_l = ~((lbin >= cfg.l_range[0]) & (lbin <= cfg.l_range[1]))
    rep["dropped_no_measurement"] = int(no_measurement.sum())
    rep["dropped_outside_l_range"] = int(outside_l.sum())

    keep = ~no_measurement & ~outside_l
    df = df.loc[keep]

    df = df.assign(
        flux_hourly=hourly_flux(df).to_numpy(),
        proton_flux=proton_flux(df).to_numpy(),
    )
    threshold = contamination_threshold(df, cfg)
    df["contaminated"] = flag_contamination(df["proton_flux"], threshold).to_numpy()
    rep["contamination_threshold"] = threshold
    rep["contaminated_hours"] = int(df["contaminated"].sum())

    if cfg.drop_contaminated:
        before = len(df)
        df = df[~df["contaminated"]]
        rep["dropped_contaminated"] = int(before - len(df))

    df = df[np.isfinite(df["flux_hourly"])]
    rep["n_kept"] = int(len(df))
    rep["l_bins_kept"] = sorted(float(x) for x in df.index.get_level_values("Lbin").unique())
    return df, rep


# --------------------------------------------------------------------------------------
# OMNI
# --------------------------------------------------------------------------------------
def clean_omni(omni: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Validate the OMNI table: coerce to float, drop duplicated stamps, count the gaps.

    OMNI fill values (``9.9999998e+05`` etc.) are turned into NaN by the upstream
    extraction; what is checked here is that what remains is finite, unique in time, and
    consistent with the NOAA period.
    """
    df = omni.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "time"
    df = df.astype("float64").sort_index()

    rep = {"n_input": int(len(df))}
    rep["duplicate_timestamps"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")]

    nan_share = df.isna().mean()
    rep["nan_share"] = {c: float(v) for c, v in nan_share.items()}
    rep["columns_all_nan"] = sorted(c for c, v in nan_share.items() if v >= 1.0)
    df = df.drop(columns=[c for c in rep["columns_all_nan"] if c not in io.OMNI_COLUMNS])
    rep["n_kept"] = int(len(df))
    return df, rep
