"""
cleaner.py
==========
Step 1, first bullet of the SP519 project:
"Clean and synchronize the data (missing values, invalid values, outliers, time stamps)."

Nothing else is done here (no L selection, no averaging, no scaling, no windowing).

    as_dataframe   time stamps : any loader output -> float64 DataFrame, naive-UTC sorted unique index
    clean_raw      invalid / missing values : fill values -> NaN, invalid L, in-flight calibration,
                   rows without target or L removed; outliers -> NaN (IQR rule, per L bin)
    synchronize    put other series (OMNI: V, Bz, F10.7...) on the NOAA time stamps
    clean          as_dataframe + clean_raw in one call, returns (df, report)
    save_cleaned   write the result to data/cleaned, only if that folder is empty
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TARGET = "mep_pro_tel90_flux_p6"
L_COL = "L_IGRF"
P6_COUNT_STEP = 243.9024  # flux of one count in the P6 channel [#/cm2-s-str]
L_VALID_RANGE = (1.0, 20.0)  # L outside is a fill value (-1) or an open field line

RAW_COLUMNS = (TARGET, "mep_pro_tel0_flux_p6", "mep_ele_tel90_flux_e3", L_COL, "MLT",
               "mag_lat_sat", "lat", "lon", "alt", "Btot_sat", "mep_IFC_on")


@dataclass
class CleanConfig:
    target: str = TARGET
    l_col: str = L_COL
    drop_ifc: bool = True  # drop samples taken during in-flight calibration (flag == 1)
    outlier_k: float | None = 3.0  # IQR multiplier on log10 flux (None = keep outliers)
    outlier_l_bin: float = 1.0  # outliers are searched inside L bins of this width
    outlier_min_group: int = 50  # minimum positive samples in a bin to apply the rule


# --------------------------------------------------------------------------------------
# Time stamps
# --------------------------------------------------------------------------------------
def _to_datetime_index(t) -> pd.DatetimeIndex:
    t = pd.Series(np.asarray(t))
    if pd.api.types.is_datetime64_any_dtype(t):
        return pd.DatetimeIndex(t)
    v = t.astype("int64")  # NOAA 'time' is uint64 milliseconds since 1970
    m = v.abs().max()
    unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
    return pd.DatetimeIndex(pd.to_datetime(v, unit=unit))


def as_dataframe(data, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Loader chunk -> float64 DataFrame indexed by naive-UTC, sorted, unique time stamps.

    Accepts a DataFrame, an xarray.Dataset, a dict of arrays or a NetCDF path. For xarray the
    variables are read one by one (``columns`` defaults to ``RAW_COLUMNS``): in the NCEI files
    every variable is also its own dimension, so ``Dataset.to_dataframe()`` would explode.
    """
    if isinstance(data, (str, os.PathLike)):
        import xarray as xr

        data = xr.open_dataset(data, decode_times=False)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            df = df[keep + (["time"] if "time" in df.columns and "time" not in keep else [])]
    elif hasattr(data, "variables") and hasattr(data, "attrs"):  # xarray.Dataset
        names = [c for c in (columns or RAW_COLUMNS) if c in data.variables]
        df = pd.DataFrame({n: np.asarray(data[n].values) for n in names})
        if "time" in data.variables:
            df["time"] = np.asarray(data["time"].values)
    else:
        df = pd.DataFrame({k: np.asarray(v) for k, v in dict(data).items()})
        if columns is not None:
            df = df[[c for c in [*columns, "time"] if c in df.columns]]

    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "time" in df.columns:
        idx = _to_datetime_index(df.pop("time").to_numpy())
    else:
        raise KeyError("no DatetimeIndex and no 'time' column in the data")
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = pd.DatetimeIndex(idx, name="time")
    return df.astype("float64")


# --------------------------------------------------------------------------------------
# Invalid / missing values and outliers
# --------------------------------------------------------------------------------------
def mask_outliers(df: pd.DataFrame, cfg: CleanConfig) -> pd.Series:
    """Boolean mask of outliers of the target: IQR rule on log10 flux, per L bin.

    Zero-count samples are real values and are neither used for the quartiles nor flagged.
    Bins with fewer than ``outlier_min_group`` positive samples are left untouched.
    """
    out = pd.Series(False, index=df.index)
    if cfg.outlier_k is None:
        return out
    flux = df[cfg.target]
    logv = np.log10(flux.where(flux > 0))
    lbin = np.floor(df[cfg.l_col] / cfg.outlier_l_bin)
    g = logv.groupby(lbin)
    q = g.quantile([0.25, 0.75]).unstack()
    n = g.count()
    ok = n >= cfg.outlier_min_group
    q1, q3 = lbin.map(q[0.25].where(ok)), lbin.map(q[0.75].where(ok))
    iqr = q3 - q1
    return ((logv < q1 - cfg.outlier_k * iqr) | (logv > q3 + cfg.outlier_k * iqr)).fillna(False)


def clean_raw(df: pd.DataFrame, cfg: CleanConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Remove invalid and missing values, flag outliers. Returns ``(clean_df, report)``."""
    cfg = cfg or CleanConfig()
    for c in (cfg.target, cfg.l_col):
        if c not in df.columns:
            raise KeyError(f"column {c!r} missing (got {list(df.columns)})")
    df = df.copy()
    rep = {"n_raw": len(df),
           "duplicate_timestamps": int(df.index.duplicated().sum()),
           "unsorted_timestamps": int((df.index.to_series().diff() < pd.Timedelta(0)).sum())}
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # fill values (-999 in several MEPED channels) are negative fluxes -> NaN
    for c in [c for c in df.columns if "flux" in c]:
        bad = df[c] < 0
        if bad.any():
            rep[f"fill_values_{c}"] = int(bad.sum())
            df.loc[bad, c] = np.nan

    # invalid L: -1 fill value, open field lines (L ~ 100)
    bad_l = ~df[cfg.l_col].between(*L_VALID_RANGE)
    rep["invalid_L"] = int(bad_l.sum())
    df.loc[bad_l, cfg.l_col] = np.nan

    # in-flight calibration (flag 1 = on; -1 is "not applicable" and kept)
    if cfg.drop_ifc and "mep_IFC_on" in df.columns:
        ifc = df["mep_IFC_on"] == 1
        rep["ifc_on"] = int(ifc.sum())
        df = df[~ifc]

    # missing values: a sample is useless without target or L
    miss = df[[cfg.target, cfg.l_col]].isna().any(axis=1)
    rep["rows_removed_missing"] = int(miss.sum())
    df = df[~miss]

    # outliers -> NaN on the target, then drop those rows
    out = mask_outliers(df, cfg)
    rep["outliers"] = int(out.sum())
    df = df[~out]
    rep["n_clean"] = len(df)
    return df, rep


def clean(data, cfg: CleanConfig | None = None, columns=None) -> tuple[pd.DataFrame, dict]:
    """``as_dataframe`` + ``clean_raw``."""
    return clean_raw(as_dataframe(data, columns), cfg)


# --------------------------------------------------------------------------------------
# Synchronisation
# --------------------------------------------------------------------------------------
def synchronize(base: pd.DataFrame, *others: pd.DataFrame, tolerance: str = "1h") -> pd.DataFrame:
    """Attach other series (e.g. OMNI V, Bz, F10.7) to the NOAA time stamps.

    Each value is the nearest one in time, accepted only within ``tolerance`` (else NaN).
    ``others`` go through ``as_dataframe`` so they can come in any format / time unit.
    """
    out = base.copy()
    tol = pd.Timedelta(tolerance)
    for o in others:
        o = as_dataframe(o)
        o = o[~o.index.duplicated(keep="first")].sort_index()
        o = o.drop(columns=[c for c in o.columns if c in out.columns])
        out = out.join(o.reindex(out.index, method="nearest", tolerance=tol))
    return out


# --------------------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------------------
def save_cleaned(df: pd.DataFrame, outdir: str = "data/cleaned", name: str | None = None) -> str | None:
    """Save the cleaned data in ``outdir`` (created if needed) ONLY if the folder is empty.

    Parquet if pyarrow/fastparquet is installed, otherwise gzipped CSV. Returns the written path,
    or None when the folder already contains files (nothing is overwritten).
    """
    os.makedirs(outdir, exist_ok=True)
    if os.listdir(outdir):
        print(f"{outdir} is not empty: nothing saved")
        return None
    name = name or f"cleaned_{df.index[0]:%Y%m%d}_{df.index[-1]:%Y%m%d}"
    try:
        path = os.path.join(outdir, name + ".parquet")
        df.to_parquet(path)
    except ImportError:
        path = os.path.join(outdir, name + ".csv.gz")
        df.to_csv(path)
    print(f"saved {len(df)} rows to {path}")
    return path


# --------------------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from loader import iter_data

    data = iter_data().__next__()  # Load the first chunk of data
    cfg = CleanConfig()

    df = as_dataframe(data)
    print(f"raw: {len(df)} samples, {df.index[0]} -> {df.index[-1]}")
    clean_df, rep = clean_raw(df, cfg)
    print("\ncleaning report:")
    for k, v in rep.items():
        print(f"  {k:<28}{v:>8}")
    print(f"\nremaining NaN in target / L: {int(clean_df[[cfg.target, cfg.l_col]].isna().sum().sum())}")
    print(f"time stamps sorted & unique: {clean_df.index.is_monotonic_increasing and clean_df.index.is_unique}")

    # Synchronisation demo with a SYNTHETIC hourly series (replace it by OMNI V, Bz, F10.7).
    fake = pd.DataFrame({"V": 400 + 50 * np.sin(np.arange(30) / 4)},
                        index=pd.date_range(df.index[0].floor("h"), periods=30, freq="1h"))
    sync = synchronize(clean_df, fake, tolerance="1h")
    print("\nafter synchronize:\n", sync[[cfg.target, cfg.l_col, "V"]].head(3))

    save_cleaned(clean_df)  # data/cleaned/, only if the folder is empty