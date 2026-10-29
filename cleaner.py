"""
cleaner.py
==========
Step 1: "Clean and synchronize the data (missing values, invalid values, outliers, time stamps)".

Pipeline:  loader.iter_data -> clean_dataset -> data/cleaned/cleaned.parquet
           -> loader.load_cleaned_data -> biner.bin_mean -> visualizer

``clean_dataset`` reads the files twice, one chunk at a time, so the dataset is never fully in memory:
  pass 1  histogram of log10(flux) over all chunks      -> global outlier bounds (IQR rule)
  pass 2  clean each chunk, keep L_RANGE, remove outliers, attach the solar parameters,
          append the chunk to ONE parquet file (the one ``loader.load_cleaned_data`` reads)
Nothing is written if ``data/cleaned`` is not empty.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

TARGET = "mep_pro_tel90_flux_p6"
L_COL = "L_IGRF"
L_RANGE = (3.5, 5.0)  # region predicted, given by the L module
OUTLIER_K = 3.0  # outlier = outside [Q1 - k*IQR, Q3 + k*IQR] on log10(flux)
P6_COUNT_STEP = 243.9024  # flux of one count in the P6 channel [#/cm2-s-str]
RAW_COLUMNS = (TARGET, L_COL, "mep_IFC_on")  # the only variables read from the 158 of a file

LOG_EDGES = np.arange(0.0, 8.0 + 1e-9, 0.005)  # histogram bins of log10(flux), 0.005 resolution


@dataclass
class CleanConfig:
    target: str = TARGET
    l_col: str = L_COL
    drop_ifc: bool = True  # drop samples taken during in-flight calibration
    outlier_k: float | None = OUTLIER_K  # None = keep outliers
    solar_fill_values: dict = field(default_factory=dict)  # e.g. {"V": 99999.9, "Bz": 999.99}
    solar_tolerance: str = "1h"  # max time distance between a NOAA sample and a solar value


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
    """Loader chunk (or any DataFrame / dict / xarray / path) -> float64 DataFrame
    indexed by naive-UTC time stamps. ``columns`` restricts the variables kept."""
    if isinstance(data, (str, os.PathLike)):
        import xarray as xr

        data = xr.open_dataset(data, decode_times=False)
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif hasattr(data, "variables"):  # xarray.Dataset (read variable by variable, see loader)
        names = [c for c in (columns or RAW_COLUMNS) if c in data.variables] + ["time"]
        df = pd.DataFrame({n: np.asarray(data[n].values) for n in dict.fromkeys(names)})
    else:
        df = pd.DataFrame({k: np.asarray(v) for k, v in dict(data).items()})
    if columns is not None:
        df = df[[c for c in dict.fromkeys([*columns, "time"]) if c in df.columns]]

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
# Invalid / missing values (one chunk)
# --------------------------------------------------------------------------------------
def clean_chunk(chunk, cfg: CleanConfig, l_range=L_RANGE) -> tuple[pd.DataFrame, dict]:
    """Clean one loader chunk and keep ``l_range``. Returns ``(df[target, L], report)``."""
    df = as_dataframe(chunk, RAW_COLUMNS)
    rep = {"n_raw": len(df), "duplicate_timestamps": int(df.index.duplicated().sum())}
    df = df[~df.index.duplicated(keep="first")].sort_index()

    bad = df[cfg.target] < 0  # fill values (-999) -> missing
    rep["fill_values"] = int(bad.sum())
    df.loc[bad, cfg.target] = np.nan

    if cfg.drop_ifc and "mep_IFC_on" in df:  # in-flight calibration (1 = on; -1 = n/a, kept)
        ifc = df["mep_IFC_on"] == 1
        rep["ifc_on"] = int(ifc.sum())
        df = df[~ifc]

    rep["missing_target"] = int(df[cfg.target].isna().sum())
    keep = df[cfg.target].notna() & df[cfg.l_col].between(*l_range)  # invalid L (-1, >20) falls out too
    rep["removed_missing_or_outside_L"] = int((~keep).sum())
    return df.loc[keep, [cfg.target, cfg.l_col]], rep


# --------------------------------------------------------------------------------------
# Outliers on the whole dataset, without loading it
# --------------------------------------------------------------------------------------
def outlier_bounds(chunks, cfg: CleanConfig, l_range=L_RANGE) -> tuple[float, float]:
    """Global (low, high) bounds of log10(flux): pass 1 over all chunks.

    ``chunks`` is a callable returning a fresh iterator of chunks (e.g. ``loader.iter_data``).
    Only a histogram is kept in memory; quartiles are read from it (resolution 0.005 in log10).
    Zero-count samples are real values: ignored by the quartiles and never flagged.
    """
    hist = np.zeros(len(LOG_EDGES) - 1)
    for chunk in chunks():
        df, _ = clean_chunk(chunk, cfg, l_range)
        flux = df[cfg.target]
        hist += np.histogram(np.log10(flux[flux > 0]), LOG_EDGES)[0]
    if hist.sum() == 0:
        raise ValueError("no positive flux in the selected L range")
    cdf = hist.cumsum() / hist.sum()
    centers = (LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2
    q1, q3 = (centers[np.searchsorted(cdf, p)] for p in (0.25, 0.75))
    return q1 - cfg.outlier_k * (q3 - q1), q3 + cfg.outlier_k * (q3 - q1)


# --------------------------------------------------------------------------------------
# Synchronisation with the solar parameters
# --------------------------------------------------------------------------------------
def clean_solar(solar, fill_values: dict | None = None) -> pd.DataFrame:
    """Solar parameters (V, Bz, F10.7...) -> DataFrame with a UTC index; fill values -> NaN."""
    df = as_dataframe(solar)
    for col, val in (fill_values or {}).items():
        if col in df:
            df.loc[np.isclose(df[col], val), col] = np.nan
    return df[~df.index.duplicated(keep="first")].sort_index()


def synchronize(base: pd.DataFrame, other: pd.DataFrame, tolerance: str = "1h") -> pd.DataFrame:
    """Attach to each time stamp of ``base`` the nearest value of ``other`` (NaN beyond ``tolerance``)."""
    other = other.drop(columns=[c for c in other.columns if c in base.columns])
    return base.join(other.reindex(base.index, method="nearest", tolerance=pd.Timedelta(tolerance)))


# --------------------------------------------------------------------------------------
# Whole dataset
# --------------------------------------------------------------------------------------
def cleaned_dir() -> str:
    """Folder read by ``loader.load_cleaned_data``."""
    from loader import CLEANED_DATA_DIR, DATA_DIR

    return os.path.join(DATA_DIR, CLEANED_DATA_DIR)


def clean_dataset(chunks, cfg: CleanConfig | None = None, solar=None, l_range=L_RANGE,
                  outdir: str | None = None) -> dict | None:
    """Clean the whole dataset chunk by chunk and write ``outdir/cleaned.parquet``.

    ``chunks`` : callable returning an iterator of chunks (``loader.iter_data``); read twice
    ``solar``  : optional DataFrame of solar parameters, attached to the NOAA time stamps
    Returns the report (summed over chunks), or None if ``outdir`` is not empty (nothing done).
    Needs pyarrow.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg, outdir = cfg or CleanConfig(), outdir or cleaned_dir()
    os.makedirs(outdir, exist_ok=True)
    if os.listdir(outdir):
        print(f"{outdir} is not empty: nothing done")
        return None

    low, high = outlier_bounds(chunks, cfg, l_range) if cfg.outlier_k is not None else (-np.inf, np.inf)
    if solar is not None:
        solar = clean_solar(solar, cfg.solar_fill_values)

    path, writer, last, total = os.path.join(outdir, "cleaned.parquet"), None, None, {}
    try:
        for chunk in chunks():
            df, rep = clean_chunk(chunk, cfg, l_range)
            flux = df[cfg.target]
            logf = np.log10(flux.where(flux > 0))  # zeros -> NaN -> never flagged
            out = (logf < low) | (logf > high)
            rep["outliers"] = int(out.sum())
            df = df[~out]
            if solar is not None:
                df = synchronize(df, solar, cfg.solar_tolerance)
            if last is not None:  # same time stamp at the border of two files
                rep["duplicates_between_files"] = int((df.index <= last).sum())
                df = df[df.index > last]
            if df.empty:
                continue
            last = df.index[-1]
            table = pa.Table.from_pandas(df)
            writer = writer or pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            rep["n_clean"] = len(df)
            for k, v in rep.items():
                total[k] = total.get(k, 0) + v
        if writer is None:
            raise ValueError("no data left after cleaning")
    except BaseException:
        if writer:
            writer.close()
            writer = None
        if os.path.exists(path):
            os.remove(path)  # never leave a half-written file: it would block the next run
        raise
    writer.close()
    total.update(outlier_low_log10=float(low), outlier_high_log10=float(high))
    return total


# --------------------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from loader import iter_data, load_cleaned_data

    try:  # solar parameters: no function for them in the loader yet
        from loader import load_solar_data

        solar = load_solar_data()
    except ImportError:
        solar = None
        print("loader.load_solar_data not found: the solar parameters are NOT attached\n")

    report = clean_dataset(iter_data, solar=solar)  # pass the function: it is read twice
    if report:
        print("cleaning report (summed over all files):")
        for k, v in report.items():
            print(f"  {k:<32}{v:>12.4g}" if isinstance(v, float) else f"  {k:<32}{v:>12}")

        cleaned = load_cleaned_data()  # what the next modules will read
        print(f"\ncleaned data: {cleaned.shape}, columns {list(cleaned.columns)}")
        print("L in range:", cleaned[L_COL].between(*L_RANGE).all(),
              "| no NaN in flux/L:", not cleaned[[TARGET, L_COL]].isna().any().any(),
              "| time sorted & unique:", cleaned.index.is_monotonic_increasing and cleaned.index.is_unique)