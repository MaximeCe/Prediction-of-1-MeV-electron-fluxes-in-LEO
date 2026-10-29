"""One call: the two hourly sources -> the table that Step 2 consumes.

    from sp519.config import Config
    from sp519 import pipeline

    result = pipeline.run(Config())
    result["table"]     # long table, indexed by (Lbin, time_bin)
    result["summary"]   # headline numbers, including the 95th-percentile threshold
    result["report"]    # JSON-serialisable trace of what was cleaned and why
"""
from __future__ import annotations

import dataclasses
import json
import os

import pandas as pd

from . import aggregate, clean, io, qc
from .config import Config


def _config_dict(cfg: Config) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()}


def _write(path: str, frame: pd.DataFrame) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    frame.to_parquet(path)


def run(cfg: Config | None = None) -> dict:
    """Run the whole preprocessing and (by default) write the table and its report.

    Returns a dict with ``table`` (long), ``wide``, ``summary`` and ``report``.
    """
    cfg = cfg or Config()

    noaa_raw = io.load_noaa_hourly(cfg.noaa_path)
    omni_raw = io.load_omni(cfg.omni_path)

    noaa, noaa_report = clean.clean_noaa(noaa_raw, cfg)
    omni, omni_report = clean.clean_omni(omni_raw, cfg)
    if noaa.empty:
        raise ValueError(
            f"no NOAA sample left in L range {cfg.l_range}: check the input file "
            f"and the L bin grid ({cfg.l_bin_step})"
        )

    long = aggregate.to_long_table(noaa, omni, cfg)
    wide = aggregate.to_wide_table(long)
    summary = qc.summary(long, cfg)

    report = {
        "config": _config_dict(cfg),
        "cleaning": noaa_report,
        "omni": omni_report,
        "summary": summary,
    }

    if cfg.out_path:
        _write(cfg.out_path, long.reset_index())
    if cfg.report_path:
        parent = os.path.dirname(cfg.report_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(cfg.report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    return {"table": long, "wide": wide, "summary": summary, "report": report}
