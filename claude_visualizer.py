"""
visualizer.py
=============
Visual and statistical exploration of NOAA-15 / SEM-2 data (1 MeV electron flux proxy).

The first chunk of data is loaded with ``loader.iter_data`` (see ``load_data``), then the helpers
below produce the statistics and figures needed for Step 1 of the SP519 project.

Every function accepts the raw 2 s DataFrame built by ``to_frame`` (any loader output is
converted: xarray.Dataset, DataFrame, dict, path), so they can also be used on several
concatenated chunks.

Statistics                                  Figures (all return a matplotlib Figure)
-----------------------------------------   ---------------------------------------------------
list_variables   catalogue of variables     plot_timeseries         flux / L / extra series vs time
quality_report   NaN, fill values, zeros    plot_flux_vs_L          where the belts are (L choice)
sampling_summary cadence, gaps, L coverage  plot_flux_distribution  histogram + 95th-pct threshold
describe_flux    descriptive statistics     plot_orbit_map          flux along the orbit (SAA!)
event_threshold  95th-percentile threshold  plot_correlation        correlation heatmap
outlier_summary  IQR-based outliers         plot_coverage           samples per bin in the L range
correlation_matrix                          plot_binned             output of cleaner.aggregate
                                            plot_all                everything, saved as PNG
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from claude_cleaner import L_COL, L_VALID_RANGE, P6_COUNT_STEP, TARGET, as_dataframe

FLUX_FLOOR = P6_COUNT_STEP / 2  # a "0 count" sample is drawn at half a count on log axes
DEFAULT_CORR_COLS = (TARGET, "mep_pro_tel0_flux_p6", "mep_ele_tel90_flux_e3", L_COL, "MLT",
                     "mag_lat_sat", "Btot_sat", "alt")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_data():
    """First chunk returned by the project loader (raw object: Dataset / DataFrame / ...)."""
    from loader import iter_data

    data = iter_data().__next__()  # Load the first chunk of data
    return data


def to_frame(data, columns=None) -> pd.DataFrame:
    """Loader output -> float64 DataFrame with a UTC DatetimeIndex (see cleaner.as_dataframe)."""
    return as_dataframe(data, columns)


def list_variables(data, pattern: str | None = None) -> pd.DataFrame:
    """Catalogue (name, long_name, units) of the variables; ``pattern`` filters name/long_name.

    Example: ``list_variables(data, "electron")``. Only xarray objects carry the descriptions.
    """
    if hasattr(data, "variables"):
        rows = [(n, str(v.attrs.get("long_name", "")), str(v.attrs.get("units", "")))
                for n, v in data.variables.items()]
    else:
        rows = [(c, "", "") for c in pd.DataFrame(data).columns]
    out = pd.DataFrame(rows, columns=["name", "long_name", "units"]).set_index("name")
    if pattern:
        hit = out.index.str.contains(pattern, case=False, regex=False) | \
            out["long_name"].str.contains(pattern, case=False, regex=False)
        out = out[hit]
    return out


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------
def _valid(df: pd.DataFrame) -> pd.DataFrame:
    """Copy where fill values become NaN (negative fluxes, invalid L). No row is dropped."""
    d = df.copy()
    for c in d.columns:
        if "flux" in c:
            d.loc[d[c] < 0, c] = np.nan
    if L_COL in d.columns:
        d.loc[~d[L_COL].between(*L_VALID_RANGE), L_COL] = np.nan
    return d


def _in_L(df: pd.DataFrame, l_range) -> pd.DataFrame:
    d = _valid(df)
    return d if l_range is None else d[d[L_COL].between(*l_range)]


def _log(x):
    """log10 of a flux, with 0-count samples placed at half a count (NaN stays NaN)."""
    return np.log10(np.maximum(x, FLUX_FLOOR))


def _subsample(d: pd.DataFrame, max_points: int) -> pd.DataFrame:
    return d.iloc[:: max(1, len(d) // max_points)]


def _need(df: pd.DataFrame, *cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"missing column(s) {miss}; available: {list(df.columns)}")


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column counts of NaN, negative values, zeros, plus min / max.

    Negative values are fill values (-999 in several MEPED channels, -1 in L_IGRF) for the flux
    and L columns; they are legitimate for latitudes. Zeros in the flux are real "0 count" samples.
    """
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({"column": c, "n": len(s), "nan": int(s.isna().sum()), "negative": int((s < 0).sum()),
                     "zeros": int((s == 0).sum()), "pct_zero": 100 * float((s == 0).mean()),
                     "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows).set_index("column")


def sampling_summary(df: pd.DataFrame, l_range=(3.0, 7.0)) -> dict:
    """Time span, cadence, gaps and the share of samples inside ``l_range``."""
    dt = df.index.to_series().diff().dt.total_seconds().dropna()
    cadence = float(dt.median()) if len(dt) else np.nan
    gaps = dt[dt > 2 * cadence] if len(dt) else dt
    out = {"start": df.index[0], "end": df.index[-1], "n_samples": len(df), "median_cadence_s": cadence,
           "n_gaps": int(len(gaps)), "longest_gap_s": float(gaps.max()) if len(gaps) else 0.0}
    if L_COL in df.columns and l_range is not None:
        out["samples_in_L_range"] = int(len(_in_L(df, l_range)))
        out["pct_in_L_range"] = 100 * out["samples_in_L_range"] / len(df)
    return out


def describe_flux(df: pd.DataFrame, l_range=(3.0, 7.0), col: str = TARGET) -> pd.DataFrame:
    """Descriptive statistics of the flux, linear and log10, restricted to ``l_range``."""
    _need(df, col, L_COL)
    s = _in_L(df, l_range)[col].dropna()
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    out = {}
    for name, x in (("linear", s), ("log10", _log(s))):
        d = x.describe(percentiles=q)
        d["skew"], d["kurtosis"] = x.skew(), x.kurt()
        out[name] = d
    res = pd.DataFrame(out)
    res.loc["pct_zero"] = [100 * float((s == 0).mean()), np.nan]
    return res


def event_threshold(df: pd.DataFrame, q: float = 95.0, l_range=(3.0, 7.0), freq: str | None = None,
                    col: str = TARGET) -> dict:
    """q-th percentile of the flux inside ``l_range`` -> definition of an "event".

    With ``freq`` (e.g. "12h") the percentile is taken on the bin means, as in the project;
    otherwise on the raw samples. In a forecasting experiment compute it on the training period.
    """
    _need(df, col, L_COL)
    s = _in_L(df, l_range)[col]
    if freq:
        s = s.resample(freq).mean()
    s = s.dropna()
    thr = float(np.percentile(s, q))
    return {"q": q, "threshold": thr, "log10": float(_log(thr)), "n": int(len(s)),
            "n_events": int((s > thr).sum()), "fraction_events": float((s > thr).mean()), "freq": freq}


def outlier_summary(df: pd.DataFrame, l_range=(3.0, 7.0), k: float = 3.0, freq: str | None = None,
                    col: str = TARGET, top: int = 10) -> tuple[dict, pd.DataFrame]:
    """IQR rule on log10 flux: outside [Q1 - k*IQR, Q3 + k*IQR]. (k=1.5 mild, k=3 extreme.)

    Returns ``(summary, top_outliers)``. Outliers are candidates to inspect (SPE, SAA, instrument
    glitches), not values to delete blindly: in this problem the highest fluxes are the events.
    """
    s = _in_L(df, l_range)[col]
    if freq:
        s = s.resample(freq).mean()
    x = _log(s.dropna())
    q1, q3 = x.quantile([0.25, 0.75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    out_mask = (x < lo) | (x > hi)
    summary = {"k": k, "lower_log10": float(lo), "upper_log10": float(hi), "n": int(len(x)),
               "n_low": int((x < lo).sum()), "n_high": int((x > hi).sum()),
               "pct_outliers": 100 * float(out_mask.mean())}
    worst = x[out_mask].reindex(x[out_mask].abs().sort_values(ascending=False).index)[:top]
    return summary, worst.rename("log10_flux").to_frame()


def correlation_matrix(df: pd.DataFrame, cols=None, l_range=(3.0, 7.0), method: str = "spearman",
                       freq: str | None = None) -> pd.DataFrame:
    """Correlation between columns (flux columns in log10). Spearman by default (heavy tails).

    Pass solar-wind columns (V, Bz, F10.7...) in ``cols`` once they are merged in ``df``.
    """
    d = _in_L(df, l_range)
    cols = [c for c in (cols or DEFAULT_CORR_COLS) if c in d.columns]
    if freq:
        d = d[cols].resample(freq).mean()
    x = d[cols].copy()
    for c in cols:
        if "flux" in c:
            x[c] = _log(x[c])
    return x.corr(method=method)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_timeseries(df: pd.DataFrame, l_range=(3.0, 7.0), extra_cols=(), resample: str | None = None,
                    max_points: int = 200_000):
    """Flux (in ``l_range``), L-shell and optional extra series (solar wind...) vs time."""
    _need(df, TARGET, L_COL)
    d = _valid(df)
    ex = [c for c in extra_cols if c in d.columns]
    fig, axes = plt.subplots(2 + len(ex), 1, figsize=(11, 2.7 * (2 + len(ex))), sharex=True)

    band = d[d[L_COL].between(*l_range)]
    sub = _subsample(band, max_points)
    axes[0].plot(sub.index, _log(sub[TARGET]), ".", ms=1.5, alpha=0.35, color="C0", label="2 s samples")
    if resample:
        m = band[TARGET].resample(resample).mean()
        axes[0].plot(m.index, _log(m), "-o", ms=3, color="C3", label=f"{resample} mean")
    axes[0].set_ylabel("log10 P6 flux\n[#/cm²/s/sr]")
    axes[0].set_title(f"{TARGET}, L in {l_range}  (0-count samples drawn at half a count)")
    axes[0].legend(loc="upper right", fontsize=8)

    sl = _subsample(d[d[L_COL].between(1, 10)], max_points)
    axes[1].plot(sl.index, sl[L_COL], ".", ms=1, color="C2")
    axes[1].axhspan(*l_range, color="C1", alpha=0.2, label="selected L range")
    axes[1].set_ylabel("L (IGRF)"); axes[1].set_ylim(1, 10); axes[1].legend(loc="upper right", fontsize=8)

    for ax, c in zip(axes[2:], ex):
        ax.plot(d.index, d[c], lw=0.8); ax.set_ylabel(c)
    axes[-1].set_xlabel("time (UTC)")
    fig.tight_layout()
    return fig


def plot_flux_vs_L(df: pd.DataFrame, l_range=(3.0, 7.0), l_max: float = 10.0):
    """2-D histogram of log10 flux vs L, with the mean profile: use it to choose the L range.

    Below L~2.5 the P6 channel mostly counts trapped *protons* (inner belt / South Atlantic Anomaly).
    """
    _need(df, TARGET, L_COL)
    d = _valid(df)
    d = d[d[L_COL].between(1, l_max)].dropna(subset=[TARGET])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hb = ax.hexbin(d[L_COL], _log(d[TARGET]), gridsize=(90, 35), bins="log", mincnt=1, cmap="magma")
    fig.colorbar(hb, ax=ax, label="samples")
    prof = d.groupby(pd.cut(d[L_COL], np.arange(1, l_max + 0.01, 0.25)), observed=True)[TARGET].mean()
    ax.plot([iv.mid for iv in prof.index], _log(prof.values), "c-", lw=2, label="mean flux per L bin")
    for i, edge in enumerate(l_range):
        ax.axvline(edge, color="w", ls="--", lw=1.2, label="selected L range" if i == 0 else None)
    ax.set_xlabel("L (IGRF)"); ax.set_ylabel("log10 P6 flux [#/cm²/s/sr]")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def plot_flux_distribution(df: pd.DataFrame, l_range=(3.0, 7.0), q: float = 95.0, freq: str | None = None):
    """Histogram of the log10 flux with the q-th percentile ("event" threshold)."""
    s = _in_L(df, l_range)[TARGET]
    if freq:
        s = s.resample(freq).mean()
    s = s.dropna()
    thr = event_threshold(df, q, l_range, freq)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(_log(s), bins=60, color="C0", alpha=0.8)
    ax.axvline(thr["log10"], color="r", ls="--", label=f"p{q:g} = {thr['threshold']:.0f} #/cm²/s/sr")
    if len(s) > 500:
        ax.set_yscale("log")
    ax.set_xlabel("log10 P6 flux"); ax.set_ylabel("count")
    what = f"{freq} means" if freq else "2 s samples"
    ax.set_title(f"Distribution of {what}, L in {l_range}  ({100 * (s == 0).mean():.0f}% are 0 count)")
    ax.legend(); fig.tight_layout()
    return fig


def plot_orbit_map(df: pd.DataFrame, max_points: int = 100_000):
    """Flux along the ground track. High values over the South Atlantic Anomaly are protons."""
    _need(df, TARGET, "lon", "lat")
    d = _subsample(_valid(df).dropna(subset=[TARGET]), max_points)
    lon = (d["lon"] + 180) % 360 - 180
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sc = ax.scatter(lon, d["lat"], c=_log(d[TARGET]), s=2, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="log10 P6 flux")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(alpha=0.3)
    ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame, cols=None, l_range=(3.0, 7.0), method: str = "spearman",
                     freq: str | None = None):
    """Annotated correlation heatmap (see ``correlation_matrix``)."""
    c = correlation_matrix(df, cols, l_range, method, freq)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(c), 0.6 + 0.8 * len(c)))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"{method} correlation")
    ax.set_title(f"L in {l_range}" + (f", {freq} means" if freq else ""))
    fig.tight_layout()
    return fig


def plot_coverage(df: pd.DataFrame, l_range=(3.0, 7.0), freq: str = "1h", min_samples: int | None = None):
    """Number of samples inside the L range per bin: shows how sparse the sampling is.

    NOAA-15 crosses the selected L band only a few minutes per pass, ~4 times per 101 min orbit.
    """
    n = _in_L(df, l_range)[TARGET].resample(freq).count()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(n.index, n.values, width=pd.Timedelta(freq) * 0.9, color="gray")
    if min_samples:
        ax.axhline(min_samples, color="r", ls="--", label=f"min_samples = {min_samples}"); ax.legend()
    ax.set_ylabel(f"samples / {freq}"); ax.set_title(f"Coverage of L in {l_range}")
    fig.tight_layout()
    return fig


def plot_binned(binned: pd.DataFrame, target: str = TARGET, threshold_log10: float | None = None):
    """Plot the output of ``cleaner.aggregate`` (log10 target and samples per bin)."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(binned.index, binned[target], "o-", ms=4)
    if threshold_log10 is not None:
        ax[0].axhline(threshold_log10, color="r", ls="--", label="event threshold"); ax[0].legend()
    ax[0].set_ylabel("log10 flux")
    ax[1].bar(binned.index, binned["n_samples"], width=(binned.index[1] - binned.index[0]) * 0.9, color="gray")
    ax[1].set_ylabel("samples / bin")
    fig.tight_layout()
    return fig


def plot_all(df: pd.DataFrame, outdir: str | None = "figures", l_range=(3.0, 7.0), q: float = 95.0,
             freq: str | None = None, extra_cols=(), resample: str | None = None) -> dict:
    """Generate every figure; save them as PNG in ``outdir`` (None = do not save)."""
    figs = {
        "timeseries": plot_timeseries(df, l_range, extra_cols, resample),
        "flux_vs_L": plot_flux_vs_L(df, l_range),
        "distribution": plot_flux_distribution(df, l_range, q, freq),
        "orbit_map": plot_orbit_map(df),
        "correlation": plot_correlation(df, l_range=l_range, freq=freq),
        "coverage": plot_coverage(df, l_range),
    }
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=130)
    return figs


# --------------------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    L_RANGE = (3.0, 7.0)  # outer belt; compare several ranges for the bonus question
    pd.set_option("display.width", 140, "display.max_columns", 20, "display.float_format", "{:.4g}".format)

    data = load_data()
    print(f"loader returned: {type(data).__module__}.{type(data).__name__}")
    if hasattr(data, "variables"):
        print("\n-- variables matching 'p6' --")
        print(list_variables(data, "p6").head(8))

    df = to_frame(data)
    print(f"\n-- data quality (raw, {len(df)} samples) --")
    print(quality_report(df))

    print("\n-- sampling --")
    for k, v in sampling_summary(df, L_RANGE).items():
        print(f"  {k:<22}{v}")

    print(f"\n-- flux statistics, L in {L_RANGE} --")
    print(describe_flux(df, L_RANGE))

    thr = event_threshold(df, 95, L_RANGE)
    print(f"\n-- 95th-percentile threshold (2 s samples): {thr['threshold']:.1f} #/cm2/s/sr "
          f"(log10 = {thr['log10']:.2f}), {thr['n_events']} / {thr['n']} samples above")
    thr30 = event_threshold(df, 95, L_RANGE, freq="30min")
    print(f"   on 30 min means (12 h would leave only 2 values for this 1-day chunk): "
          f"{thr30['threshold']:.1f}, {thr30['n_events']} / {thr30['n']} bins above")

    summ, worst = outlier_summary(df, L_RANGE, k=3.0)
    print(f"\n-- outliers (IQR, k={summ['k']}): {summ['n_high']} high / {summ['n_low']} low "
          f"({summ['pct_outliers']:.2f}% of {summ['n']}) --")
    print(worst.head(5))

    print("\n-- Spearman correlation --")
    print(correlation_matrix(df, l_range=L_RANGE).round(2))

    plot_all(df, outdir="figures", l_range=L_RANGE, freq="30min", resample="30min")
    print("\nfigures saved in ./figures")
    plt.show()