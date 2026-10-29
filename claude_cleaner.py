"""
cleaner.py
==========
Clean NOAA-15 / SEM-2 (MEPED) data so that it can be fed to a 1-D CNN.

Pipeline (every step is configurable through ``CleanConfig``)::

    raw chunk(s) -> as_dataframe -> clean_raw -> aggregate -> split -> fill + normalise -> windows
                    (UTC index)     (fill values,  (bins,      (chrono-  (mask channel,     X: (N, C, T)
                                     IFC, SPE, L)   log10)      logical)  z-score on train)  y: (N, H)

Design choices (to justify in the report)
-----------------------------------------
* The target ``mep_pro_tel90_flux_p6`` is nominally a *proton* channel (>6.9 MeV). In the
  radiation belts it responds to >~1 MeV electrons, but during Solar Proton Events (SPE) it is
  dominated by protons. ``flag_spe`` is a simple heuristic based on the polar-cap level of the
  channel; cross-check it against an official SPE list (e.g. GOES >10 MeV proton flux) and plug
  external intervals in if needed. Please also check the instrument documentation yourself.
* The flux is quantised (one count = 243.9 #/cm2-s-str), so zeros are real "0 count" values,
  not missing data. They are kept, and a floor of half a count is applied before ``log10``.
* Averaging is done on the *linear* flux inside each bin, then ``log10`` is applied.
* Gaps are interpolated only over ``max_interp_bins``; remaining gaps are set to 0 (= mean after
  z-scoring) and flagged by a binary "observed" channel. Targets are never interpolated.
* The split is chronological and done BEFORE any fitting: normalisation statistics, the event
  threshold (95th percentile) and the interpolation only ever see the training period, and the
  windows of each split never overlap another split.

Quick use::

    from cleaner import CleanConfig, prepare_cnn_dataset
    ds = prepare_cnn_dataset(chunks, CleanConfig(freq="12h"))   # chunks: DataFrame / Dataset / iterable
    ds.train["X"].shape      # (N, C, T) float32, channels first (PyTorch Conv1d)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
TARGET = "mep_pro_tel90_flux_p6"
L_COL = "L_IGRF"
P6_COUNT_STEP = 243.9024  # flux corresponding to one count in the P6 channel [#/cm2-s-str]
L_VALID_RANGE = (1.0, 20.0)  # L outside this range is a fill value (-1) or an open field line

# Columns pulled from the (158-variable) NetCDF file by default.
RAW_COLUMNS = (
    TARGET,
    "mep_pro_tel0_flux_p6",
    "mep_ele_tel90_flux_e3",
    L_COL,
    "MLT",
    "mag_lat_sat",
    "lat",
    "lon",
    "alt",
    "Btot_sat",
    "mep_IFC_on",
)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class CleanConfig:
    # --- selection -----------------------------------------------------------------
    target: str = TARGET
    l_col: str = L_COL
    l_range: tuple[float, float] = (3.0, 7.0)  # outer belt; to be justified / compared
    extra_flux_cols: tuple[str, ...] = ()  # more flux channels (log10 applied)
    extra_linear_cols: tuple[str, ...] = ()  # other channels, e.g. V, Bz, F10.7 (z-score only)
    use_l_channel: bool = True  # add the mean L of each bin as an input channel
    # --- raw cleaning --------------------------------------------------------------
    flux_floor: float = P6_COUNT_STEP / 2  # floor before log10 (half a count for P6)
    drop_ifc: bool = True  # drop samples taken during in-flight calibration
    spe_threshold: float | None = 1000.0  # median polar-cap flux above which a block is an SPE
    spe_polar_maglat: float = 72.0  # |magnetic latitude| defining the polar cap [deg]
    spe_freq: str = "3h"  # block length used to detect SPE (must divide 24 h)
    # --- aggregation / gaps --------------------------------------------------------
    freq: str = "12h"  # bin width (the project uses 12 h)
    min_samples: int = 10  # minimum raw samples per bin, otherwise the bin is NaN
    max_interp_bins: int = 2  # longest gap (in bins) that is interpolated
    # --- windows / labels ----------------------------------------------------------
    window: int = 8  # number of past bins fed to the CNN
    horizons_h: tuple[float, ...] = (12, 24, 36, 48)  # prediction horizons [hours]
    min_obs_frac: float = 0.5  # min. fraction of really observed bins in a window
    event_percentile: float = 95.0  # event threshold, computed on the training split only
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)  # train / val / test (chronological)

    # --- derived -------------------------------------------------------------------
    @property
    def flux_cols(self) -> list[str]:
        return [self.target, *self.extra_flux_cols]

    @property
    def value_cols(self) -> list[str]:
        cols = [*self.flux_cols, *self.extra_linear_cols]
        return cols + ["L_mean"] if self.use_l_channel else cols

    @property
    def channels(self) -> list[str]:
        return [*self.value_cols, "observed"]

    @property
    def raw_columns(self) -> list[str]:
        cols = [self.target, self.l_col, "mag_lat_sat", "mep_IFC_on", *self.extra_flux_cols]
        return list(dict.fromkeys(cols))

    @property
    def horizon_steps(self) -> list[int]:
        step = pd.Timedelta(self.freq)
        steps = []
        for h in self.horizons_h:
            k = pd.Timedelta(hours=h) / step
            if round(k) < 1 or abs(k - round(k)) > 1e-9:
                raise ValueError(f"horizon {h} h is not a positive multiple of freq={self.freq!r}")
            steps.append(int(round(k)))
        return steps


# --------------------------------------------------------------------------------------
# 1. Input normalisation: whatever the loader returns -> DataFrame with a UTC DatetimeIndex
# --------------------------------------------------------------------------------------
def _to_datetime_index(t) -> pd.DatetimeIndex:
    t = pd.Series(np.asarray(t))
    if pd.api.types.is_datetime64_any_dtype(t):
        idx = pd.DatetimeIndex(t)
    else:
        v = t.astype("int64")  # NOAA 'time' is uint64 milliseconds since 1970
        m = v.abs().max()
        unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
        idx = pd.DatetimeIndex(pd.to_datetime(v, unit=unit))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def as_dataframe(data, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Convert a loader chunk into a float64 DataFrame indexed by (naive, UTC) time.

    Accepts a ``pandas.DataFrame``, an ``xarray.Dataset``, a dict of arrays or a path to a
    NetCDF file. ``columns`` restricts the variables that are extracted; for xarray it defaults
    to ``RAW_COLUMNS`` because the file has 158 variables.

    Note: in the NCEI files every variable is also its own dimension, so calling
    ``Dataset.to_dataframe()`` would build a gigantic cartesian product. Variables are therefore
    read one by one through ``.values``.
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
    else:  # dict-like
        df = pd.DataFrame({k: np.asarray(v) for k, v in dict(data).items()})
        if columns is not None:
            df = df[[c for c in [*columns, "time"] if c in df.columns]]

    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "time" in df.columns:
        idx = _to_datetime_index(df.pop("time").to_numpy())
    else:
        raise KeyError("no DatetimeIndex and no 'time' column in the data")

    if idx.tz is not None:  # always work with naive UTC timestamps
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = pd.DatetimeIndex(idx, name="time")
    df = df.astype("float64")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# --------------------------------------------------------------------------------------
# 2. Raw-sample cleaning (2 s resolution)
# --------------------------------------------------------------------------------------
def flag_spe(df: pd.DataFrame, cfg: CleanConfig) -> pd.Series:
    """Boolean mask (True = suspected Solar Proton Event) at the sample level.

    Heuristic: in the polar caps (|mag. lat.| > ``spe_polar_maglat``) the quiet-time P6 level is
    ~0. If its median over a ``spe_freq`` block exceeds ``spe_threshold`` the whole block is
    flagged, because the channel then counts protons rather than electrons.
    """
    no_flag = pd.Series(False, index=df.index)
    if cfg.spe_threshold is None or "mag_lat_sat" not in df.columns:
        return no_flag
    polar = df["mag_lat_sat"].abs() > cfg.spe_polar_maglat
    if not polar.any():
        return no_flag
    med = df.loc[polar, cfg.target].resample(cfg.spe_freq).median()
    bad_blocks = med.index[med > cfg.spe_threshold]
    return pd.Series(df.index.floor(cfg.spe_freq).isin(bad_blocks), index=df.index)


def clean_raw(df: pd.DataFrame, cfg: CleanConfig) -> tuple[pd.DataFrame, dict]:
    """Remove invalid samples and keep only the selected L range.

    Returns ``(clean_df, report)``; ``report`` counts what was removed at each step.
    """
    for c in (cfg.target, cfg.l_col):
        if c not in df.columns:
            raise KeyError(f"column {c!r} missing from the data (got {list(df.columns)[:8]}...)")
    rep = {"n_raw": len(df)}
    df = df.copy()

    # fill values (-999 in several MEPED channels) are negative -> NaN
    for c in [c for c in df.columns if "flux" in c]:
        bad = df[c] < 0
        if bad.any():
            rep[f"fill_values_{c}"] = int(bad.sum())
            df.loc[bad, c] = np.nan

    # invalid L (-1 fill value, open field lines with L ~ 100)
    bad_l = ~df[cfg.l_col].between(*L_VALID_RANGE)
    rep["invalid_L"] = int(bad_l.sum())
    df.loc[bad_l, cfg.l_col] = np.nan

    # in-flight calibration (flag 1 = on; -1 is a "not applicable" fill, kept)
    if cfg.drop_ifc and "mep_IFC_on" in df.columns:
        ifc = df["mep_IFC_on"] == 1
        rep["ifc_on"] = int(ifc.sum())
        df = df[~ifc]

    # solar proton events (needs the polar-cap samples, so before the L selection)
    spe = flag_spe(df, cfg)
    rep["spe_flagged"] = int(spe.sum())
    df = df[~spe]

    # L range + valid target
    keep = df[cfg.l_col].between(*cfg.l_range) & df[cfg.target].notna()
    rep["outside_L_range_or_nan"] = int((~keep).sum())
    df = df.loc[keep, [c for c in [*cfg.flux_cols, *cfg.extra_linear_cols, cfg.l_col] if c in df.columns]]
    rep["n_kept"] = len(df)
    return df, rep


# --------------------------------------------------------------------------------------
# 3. Aggregation on a regular grid
# --------------------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Average the cleaned samples over ``cfg.freq`` bins and take log10 of the fluxes.

    Output columns: the flux columns (log10), extra linear columns, ``L_mean`` and ``n_samples``.
    Bins with fewer than ``min_samples`` samples are set to NaN (the index stays regular).
    """
    cols = [c for c in [*cfg.flux_cols, *cfg.extra_linear_cols] if c in df.columns]
    g = df[[*cols, cfg.l_col]].resample(cfg.freq)
    out = g.mean().rename(columns={cfg.l_col: "L_mean"})
    out["n_samples"] = g[cfg.target].count()
    out.loc[out["n_samples"] < cfg.min_samples, [*cols, "L_mean"]] = np.nan
    for c in cfg.flux_cols:
        if c in out.columns:
            out[c] = np.log10(np.maximum(out[c], cfg.flux_floor))  # NaN stays NaN
    return out.replace([np.inf, -np.inf], np.nan)


def join_external(binned: pd.DataFrame, external: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Add external series (e.g. OMNI V, Bz, F10.7) averaged on the same bins."""
    ext = external[list(cfg.extra_linear_cols)].resample(cfg.freq).mean()
    return binned.join(ext, how="left")


# --------------------------------------------------------------------------------------
# 4. Gap filling, split, normalisation
# --------------------------------------------------------------------------------------
def fill_gaps(binned: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Linear (time) interpolation of short gaps only; long gaps stay NaN."""
    out = binned.copy()
    cols = [c for c in cfg.value_cols if c in out.columns]
    out[cols] = out[cols].interpolate(method="time", limit=cfg.max_interp_bins, limit_area="inside")
    return out


def split_chronological(binned: pd.DataFrame, fractions=(0.7, 0.15, 0.15)):
    """Contiguous train / validation / test blocks, in time order (no shuffling)."""
    f = np.asarray(fractions, float)
    f = f / f.sum()
    n = len(binned)
    i1, i2 = int(round(n * f[0])), int(round(n * (f[0] + f[1])))
    return binned.iloc[:i1], binned.iloc[i1:i2], binned.iloc[i2:]


class Normalizer:
    """Per-column z-score. ``fit`` on the training split only."""

    def __init__(self):
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "Normalizer":
        self.mean_ = df[cols].mean().fillna(0.0)
        std = df[cols].std(ddof=0)
        self.std_ = std.where(std > 1e-12, 1.0).fillna(1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = list(self.mean_.index)
        out[cols] = (df[cols] - self.mean_) / self.std_
        return out

    def inverse(self, x, col: str):
        """Back to physical units (log10 flux for the target)."""
        return np.asarray(x) * self.std_[col] + self.mean_[col]

    def to_dict(self) -> dict:
        return {"mean": self.mean_.to_dict(), "std": self.std_.to_dict()}


# --------------------------------------------------------------------------------------
# 5. Sliding windows -> CNN tensors
# --------------------------------------------------------------------------------------
def make_windows(frame: pd.DataFrame, cfg: CleanConfig, norm: Normalizer, threshold: float) -> dict:
    """Build the CNN tensors of ONE split.

    Returns a dict with
        X            (N, C, T) float32, channels first; C = value columns + 'observed' mask
        y_reg        (N, H)    float32, normalised log10 target at t + each horizon
        y_log10      (N, H)    float32, same in physical log10 units
        y_event      (N, H)    int8,    1 if the target exceeds ``threshold``
        persistence  (N,)      float32, normalised target at the window end (naive baseline)
        t_end        DatetimeIndex (N,), timestamp of the last bin of each window
    Windows are dropped when a horizon target was not really observed or when fewer than
    ``min_obs_frac`` of the input bins were observed.
    """
    w, steps = cfg.window, cfg.horizon_steps
    hmax = max(steps)
    vcols = [c for c in cfg.value_cols if c in frame.columns]
    n_ch = len(vcols) + 1

    observed = frame[cfg.target].notna().to_numpy()
    target = frame[cfg.target].to_numpy(dtype=float)  # real values only (never interpolated)
    z = norm.transform(fill_gaps(frame, cfg))[vcols].fillna(0.0)
    x_full = np.concatenate([z.to_numpy(np.float32), observed[:, None].astype(np.float32)], axis=1)

    T = len(frame)
    if T < w + hmax:
        empty = np.empty
        return {"X": empty((0, n_ch, w), np.float32), "y_reg": empty((0, len(steps)), np.float32),
                "y_log10": empty((0, len(steps)), np.float32), "y_event": empty((0, len(steps)), np.int8),
                "persistence": empty((0,), np.float32), "t_end": frame.index[:0]}

    ends = np.arange(w - 1, T - hmax)  # last bin of each window
    starts = ends - (w - 1)
    X = sliding_window_view(x_full, w, axis=0)[starts]  # (N, C, T)
    y_log = np.stack([target[ends + h] for h in steps], axis=1)
    obs_frac = sliding_window_view(observed.astype(float), w)[starts].mean(axis=1)
    keep = ~np.isnan(y_log).any(axis=1) & (obs_frac >= cfg.min_obs_frac)

    mu, sd = norm.mean_[cfg.target], norm.std_[cfg.target]
    return {
        "X": np.ascontiguousarray(X[keep], dtype=np.float32),
        "y_reg": ((y_log - mu) / sd)[keep].astype(np.float32),
        "y_log10": y_log[keep].astype(np.float32),
        "y_event": (y_log > threshold)[keep].astype(np.int8),
        "persistence": z[cfg.target].to_numpy()[ends][keep].astype(np.float32),
        "t_end": frame.index[ends][keep],
    }


# --------------------------------------------------------------------------------------
# 6. One-call pipeline
# --------------------------------------------------------------------------------------
@dataclass
class CNNDataset:
    train: dict
    val: dict
    test: dict
    normalizer: Normalizer
    threshold_log10: float  # event threshold (percentile of the TRAIN target), log10 units
    channels: list[str]
    config: CleanConfig
    report: dict
    binned: pd.DataFrame = field(repr=False)  # full aggregated table (before fill / normalisation)

    def summary(self) -> str:
        lines = [f"channels: {self.channels}",
                 f"event threshold: log10 flux > {self.threshold_log10:.3f} "
                 f"({10 ** self.threshold_log10:.0f} #/cm2-s-str, p{self.config.event_percentile:g} of train)"]
        for name in ("train", "val", "test"):
            d = getattr(self, name)
            lines.append(f"{name:>5}: X{d['X'].shape}  y_reg{d['y_reg'].shape}  "
                         f"events/horizon={d['y_event'].sum(axis=0).tolist() if len(d['y_event']) else []}")
        return "\n".join(lines)


def _is_single_chunk(x) -> bool:
    return isinstance(x, (pd.DataFrame, str, os.PathLike, dict)) or hasattr(x, "variables")


def clean_chunks(chunks, cfg: CleanConfig) -> tuple[pd.DataFrame, dict]:
    """Clean one chunk or an iterable of chunks (one by one, to save memory) and concatenate."""
    if _is_single_chunk(chunks):
        chunks = [chunks]
    frames, total = [], {}
    for chunk in chunks:
        clean, rep = clean_raw(as_dataframe(chunk, cfg.raw_columns), cfg)
        frames.append(clean)
        for k, v in rep.items():
            total[k] = total.get(k, 0) + v
    if not frames:
        raise ValueError("no data chunk received")
    raw = pd.concat(frames).sort_index()
    return raw[~raw.index.duplicated(keep="first")], total


def prepare_cnn_dataset(chunks, cfg: CleanConfig | None = None, external: pd.DataFrame | None = None) -> CNNDataset:
    """Raw chunk(s) -> train / val / test tensors ready for a 1-D CNN.

    ``chunks``   : one chunk (DataFrame / Dataset / path) or an iterable of chunks (e.g. ``iter_data()``)
    ``external`` : optional DataFrame (DatetimeIndex) holding ``cfg.extra_linear_cols`` (solar wind...)
    """
    cfg = cfg or CleanConfig()
    _ = cfg.horizon_steps  # validate early
    raw, report = clean_chunks(chunks, cfg)
    binned = aggregate(raw, cfg)
    if external is not None:
        binned = join_external(binned, external, cfg)

    tr, va, te = split_chronological(binned, cfg.split)
    if tr[cfg.target].notna().sum() < 2:
        raise ValueError(
            f"only {tr[cfg.target].notna().sum()} observed training bins out of {len(binned)} bins "
            f"(freq={cfg.freq!r}): load more chunks or use a finer 'freq'.")
    cols = [c for c in cfg.value_cols if c in binned.columns]
    norm = Normalizer().fit(tr, cols)
    threshold = float(np.nanpercentile(tr[cfg.target], cfg.event_percentile))

    out = [make_windows(part, cfg, norm, threshold) for part in (tr, va, te)]
    return CNNDataset(*out, normalizer=norm, threshold_log10=threshold, channels=cfg.channels,
                      config=cfg, report=report, binned=binned)


# --------------------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from itertools import islice

    import matplotlib.pyplot as plt
    from loader import iter_data

    N_CHUNKS = 1  # a chunk is one day of 2 s data; increase to get meaningful train/val/test sets

    # --- Demo settings -------------------------------------------------------------------
    # With ONE day of data, 12 h bins would give only 2 samples, so the demo uses 30 min bins.
    # For the real project use freq="12h", window=8 (4 days), horizons_h=(12, 24, 36, 48).
    cfg = CleanConfig(freq="30min", min_samples=10, window=4, horizons_h=(0.5, 1.0, 1.5, 2.0),
                      split=(0.6, 0.2, 0.2), max_interp_bins=2)

    chunks = list(islice(iter_data(), N_CHUNKS))

    # 1) step by step -------------------------------------------------------------------
    df = as_dataframe(chunks[0], cfg.raw_columns)
    print(f"raw chunk: {len(df)} samples from {df.index[0]} to {df.index[-1]}")
    clean, rep = clean_raw(df, cfg)
    print("cleaning report:")
    for k, v in rep.items():
        print(f"  {k:<32}{v:>8}")
    binned = aggregate(clean, cfg)
    print(f"\nbinned table ({cfg.freq}): {binned.shape}, "
          f"{binned[cfg.target].isna().sum()} empty bins\n{binned.head(4)}\n")

    # 2) one call: everything up to the tensors -----------------------------------------
    ds = prepare_cnn_dataset(chunks, cfg)
    print(ds.summary())
    for name in ("train", "val", "test"):
        d = getattr(ds, name)
        for key in ("X", "y_reg"):
            assert np.isfinite(d[key]).all(), f"non finite values in {name}/{key}"
    print("\nOK: no NaN/inf in any tensor; dtype =", ds.train["X"].dtype)
    print("example: X[0] has shape (channels, window) =", ds.train["X"][0].shape if len(ds.train["X"]) else None)

    # 3) quick look ---------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(clean.index, np.log10(np.maximum(clean[cfg.target], cfg.flux_floor)), ".", ms=1.5, alpha=0.4,
               label="cleaned samples (2 s)")
    ax[0].plot(binned.index, binned[cfg.target], "o-", color="C3", ms=4, label=f"{cfg.freq} mean")
    ax[0].axhline(ds.threshold_log10, color="k", ls="--", lw=1, label="event threshold (train p95)")
    ax[0].set_ylabel("log10 P6 flux"); ax[0].legend(loc="upper right", fontsize=8)
    ax[1].bar(binned.index, binned["n_samples"], width=pd.Timedelta(cfg.freq) * 0.9, color="gray")
    ax[1].axhline(cfg.min_samples, color="r", ls="--", lw=1)
    ax[1].set_ylabel("samples / bin")
    fig.suptitle(f"Cleaned data, L in {cfg.l_range}")
    fig.tight_layout()
    fig.savefig("cleaner_demo.png", dpi=120)
    print("figure saved: cleaner_demo.png")
    plt.show()