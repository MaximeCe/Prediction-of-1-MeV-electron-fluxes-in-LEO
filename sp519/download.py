"""Download the raw NOAA-15 SEM-2 daily NetCDF files from the NCEI portal.

This is the *upstream* half of Step 1: the daily 2 s files are reduced offline into the
hourly table ``data/noaa15_p6_hourly_Lbins.parquet`` that ``sp519.io`` reads. Only this
download is kept in the repository, because the raw files are ~12 GB.

Two behaviours of the previous version were dangerous and are fixed here:

* it emptied the whole target directory, and its default target was ``data/`` — the same
  directory that holds the versioned parquet tables, which it would therefore delete;
* it re-downloaded everything on every run, with no way to resume.

Now the target is a dedicated ``data/noaa15_raw/`` folder, existing files are skipped unless
``overwrite`` is set, and nothing outside that folder is ever touched.
"""
from __future__ import annotations

import calendar
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import requests

RAW_DIR = os.path.join("data", "noaa15_raw")
MAX_WORKERS = 8
BASE_URL = (
    "https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/"
    "access/l1b/v01r00/{year}/noaa15/poes_n15_{date}_proc.nc"
)


def daily_files(years: Iterable[int], months: Iterable[int]) -> list[tuple[int, str, str]]:
    """``[(year, 'YYYYMMDD', 'poes_n15_YYYYMMDD_proc.nc'), ...]`` for the requested months."""
    out = []
    for year in years:
        for month in months:
            if not 1 <= month <= 12:
                raise ValueError(f"month must be between 1 and 12, got {month}")
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                date = f"{year:04d}{month:02d}{day:02d}"
                out.append((year, date, f"poes_n15_{date}_proc.nc"))
    return out


def _download_one(entry: tuple[int, str, str], data_dir: str, overwrite: bool) -> str | None:
    year, date, filename = entry
    destination = os.path.join(data_dir, filename)
    if os.path.exists(destination) and not overwrite:
        return destination
    url = BASE_URL.format(year=year, date=date)
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            tmp = destination + ".part"
            with open(tmp, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
            os.replace(tmp, destination)
        return destination
    except requests.exceptions.RequestException as exc:  # network / 404: keep going
        print(f"failed: {url} ({exc})")
        return None


def download_nc_files(years, months, data_dir: str = RAW_DIR, max_workers: int = MAX_WORKERS,
                      overwrite: bool = False) -> list[str]:
    """Download the daily files of the requested months; returns the paths actually present.

    Existing files are reused, so an interrupted download restarts where it stopped.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    os.makedirs(data_dir, exist_ok=True)
    targets = daily_files(years, months)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda e: _download_one(e, data_dir, overwrite), targets))
    done = [p for p in results if p]
    print(f"{len(done)} / {len(targets)} daily files in {data_dir}")
    return done


if __name__ == "__main__":
    download_nc_files(years=[2020], months=range(1, 13))
