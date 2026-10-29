#!/usr/bin/env python
"""SP519 preprocessing pipeline - command line entry point (Step 1).

    python pre-processing-pipeline.py                          # clean + 12 h table + report
    python pre-processing-pipeline.py --figures                # ... and the Step 1 figures
    python pre-processing-pipeline.py --l-range 3 5            # try another belt region
    python pre-processing-pipeline.py --download 2020          # fetch the raw NOAA-15 files

Reads the two hourly parquet files in ``data/`` and writes the modelling table consumed by
Step 2. All the reasoning choices live in ``sp519/config.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys

from sp519 import pipeline, plots, qc
from sp519.config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l-range", nargs=2, type=float, metavar=("LO", "HI"),
                   help="L range of the belt region to predict (default 3.0 6.0)")
    p.add_argument("--freq", default=None, help="bin width, e.g. 12h (default 12h)")
    p.add_argument("--min-coverage", type=float, default=None,
                   help="drop intervals observed less than this fraction of the time")
    p.add_argument("--event-percentile", type=float, default=None,
                   help="percentile that defines an event (default 95)")
    p.add_argument("--noaa", default=None, help="hourly NOAA-15 parquet to read")
    p.add_argument("--omni", default=None, help="hourly OMNI parquet to read")
    p.add_argument("--out", default=None,
                   help="output parquet (default data/table_12h_2020_2024.parquet)")
    p.add_argument("--figures", action="store_true", help="also write the Step 1 figures")
    p.add_argument("--figures-dir", default="figures", help="where to write the figures")
    p.add_argument("--download", nargs="+", type=int, metavar="YEAR",
                   help="download the raw NOAA-15 files for the given years and exit")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {}
    if args.l_range:
        overrides["l_range"] = tuple(args.l_range)
    if args.freq:
        overrides["freq"] = args.freq
    if args.min_coverage is not None:
        overrides["min_coverage"] = args.min_coverage
    if args.event_percentile is not None:
        overrides["event_percentile"] = args.event_percentile
    if args.noaa:
        overrides["noaa_path"] = args.noaa
    if args.omni:
        overrides["omni_path"] = args.omni
    if args.out:
        overrides["out_path"] = args.out
    return dataclasses.replace(Config(), **overrides)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.download:
        from sp519 import download

        download.download_nc_files(years=args.download, months=range(1, 13))
        return 0

    cfg = config_from_args(args)

    print("SP519 preprocessing")
    print("  NOAA-15 :", cfg.noaa_path)
    print("  OMNI    :", cfg.omni_path)
    print()

    result = pipeline.run(cfg)
    print(qc.format_summary(result["summary"]))
    print()
    print(f"written : {cfg.out_path}")
    if cfg.report_path:
        print(f"report  : {cfg.report_path}")

    print()
    print("cleaning")
    for key, value in result["report"]["cleaning"].items():
        print(f"  {key:<24}{value}")

    if args.figures:
        plots.save_all(result["table"], cfg, args.figures_dir)
        print(f"\nfigures : {args.figures_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
