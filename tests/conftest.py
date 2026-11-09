"""Shared fixtures: small synthetic NOAA-15 / OMNI tables with known numbers.

The synthetic data is built so that every expected value can be computed by hand:

* 24 hourly stamps, 2020-01-01 00:00 -> 23:00, so 12 h binning gives exactly 2 bins
  (A = 00:00-11:00, B = 12:00-23:00).
* ``Lbin`` 3.0 -> hourly flux 100, 4.5 -> 1000, 6.0 -> 400, and out-of-scope bins
  2.0 / 7.0 / 10.0 (the last one is the polar cap) that must be dropped.
* ``Lbin`` 4.5 at 03:00 has ``n = 0`` (no measurement) -> flux undefined, row dropped.
* ``Lbin`` 4.5 at 15:00 carries a solar-proton-event signature in ``omni_p1``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

HOURS = pd.date_range("2020-01-01", periods=24, freq="h", name="time")
LBINS = [2.0, 3.0, 4.5, 6.0, 7.0, 10.0]
FLUX_BY_LBIN = {3.0: 100.0, 4.5: 1000.0, 6.0: 400.0}
SAMPLES_PER_HOUR = 10
SPE_HOUR = HOURS[15]  # an SPE-contaminated hour on Lbin 4.5


@pytest.fixture
def tiny_noaa() -> pd.DataFrame:
    rows = []
    for lb in LBINS:
        flux = FLUX_BY_LBIN.get(lb, 0.0)
        for t in HOURS:
            n = SAMPLES_PER_HOUR
            if lb == 4.5 and t == HOURS[3]:
                n = 0  # no measurement in this hour
            rows.append(
                {
                    "time": t,
                    "Lbin": lb,
                    "p6_sum": flux * n,
                    "n": n,
                    "p6_max": flux if n else np.nan,
                    "omni_p1_sum": 0.05 * n,
                    "omni_p1_n": n,
                }
            )
    df = pd.DataFrame(rows)
    spe = (df["Lbin"] == 4.5) & (df["time"] == SPE_HOUR)
    df.loc[spe, "omni_p1_sum"] = 50.0  # 25 MeV protons -> 5.0 once divided by n
    return df


@pytest.fixture
def tiny_omni() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "V_kms": 400.0,
            "Bz_nT": -2.0,
            "F107_sfu": 100.0,
            "N_cm3": 5.0,
            "P_nPa": 1.0,
            "E_mVm": 0.5,
            "B_nT": 5.0,
            "Kp": 2.0,
            "Dst_nT": -10.0,
            "AE_nT": 100.0,
        },
        index=HOURS,
    )
