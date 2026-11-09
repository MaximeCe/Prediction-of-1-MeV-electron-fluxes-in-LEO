"""Reading the two hourly sources the preprocessing starts from.

The pipeline deliberately starts from *hourly* files: the 12 GB of raw 2 s NOAA-15 NetCDF
are reduced once, offline, into ``noaa15_p6_hourly_Lbins.parquet`` (see
``sp519.download``/``build_hourly`` for the raw half). Everything downstream is fast and
reproducible from the two versioned parquet files.
"""
from __future__ import annotations

import os

import pandas as pd

# Columns of the hourly NOAA-15 table, in the order produced upstream.
NOAA_COLUMNS = ("Lbin", "p6_sum", "n", "p6_max", "omni_p1_sum", "omni_p1_n")
# Solar-wind and geomagnetic parameters of the OMNI hourly table.
OMNI_COLUMNS = ("V_kms", "Bz_nT", "F107_sfu", "N_cm3", "P_nPa", "E_mVm", "B_nT",
                "Kp", "Dst_nT", "AE_nT")
POLAR_LBIN = 10.0  # L bin used as the polar-cap reference (L >= 10), not part of the belt

DEFAULT_NOAA_PATH = "data/noaa15_p6_hourly_Lbins.parquet"
DEFAULT_OMNI_PATH = "data/omni_hourly_2020_2024.parquet"


def _read_parquet(path) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_parquet(path)


def load_noaa_hourly(path=DEFAULT_NOAA_PATH) -> pd.DataFrame:
    """Hourly NOAA-15 P6 flux per L bin, indexed by UTC time (long format).

    One row = one hour x one L bin, with ``p6_sum`` and ``n`` (number of 2 s measurements),
    so that the hourly mean flux is ``p6_sum / n``. ``omni_p1_*`` are the counts of the same
    90 deg telescope in the >25 MeV proton channel, used to detect proton contamination.
    """
    df = _read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df[[c for c in NOAA_COLUMNS if c in df.columns]]


def load_omni(path=DEFAULT_OMNI_PATH) -> pd.DataFrame:
    """Hourly OMNI solar-wind and geomagnetic parameters, indexed by UTC time."""
    df = _read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "time"
    df = df.sort_index()
    return df[[c for c in OMNI_COLUMNS if c in df.columns]]
