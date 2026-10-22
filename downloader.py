import calendar
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
from typing import List
import requests

DATA_DIR = os.getenv("DATA_DIR", "data")
MAX_WORKERS = 8

def download_nc_files(
    years: List[int],
    months: List[int],
    data_dir: str = DATA_DIR,
    max_workers: int = MAX_WORKERS,
) -> None:
    """Empties the target directory and downloads NOAA POES NetCDF files.

    Args:
        years: List of years (e.g., [2020, 2021]).
        months: List of months as integers; downloads every day in each month.
        data_dir: Path to the local directory where files will be stored.
        max_workers: Maximum number of files to download concurrently.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    files_to_download = []
    for year in years:
        for month in months:
            if not 1 <= month <= 12:
                raise ValueError(f"Month must be between 1 and 12, got {month}")
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                date_str = f"{year:04d}{month:02d}{day:02d}"
                files_to_download.append(
                    (
                        year,
                        date_str,
                        f"poes_n15_{date_str}_proc.nc",
                    )
                )

    # 1. Clear or create the destination directory
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        os.makedirs(data_dir, exist_ok=True)

    base_url = (
        "https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/"
        "access/l1b/v01r00/{year}/noaa15/poes_n15_{date_str}_proc.nc"
    )

    def download_file(file_info):
        year, date_str, filename = file_info
        file_url = base_url.format(year=year, date_str=date_str)
        destination_path = os.path.join(data_dir, filename)

        print(f"Downloading: {file_url}")
        try:
            with requests.get(file_url, stream=True, timeout=30) as response:
                response.raise_for_status()

                with open(destination_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"Successfully saved to {destination_path}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_url}: {e}")

    # 2. Download daily files concurrently to reduce time spent waiting on the server.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(download_file, files_to_download))


if __name__ == "__main__":
    # Example usage:
    selected_years = [2020]
    selected_months = list(range(1, 13))  # Months from 1 to 12

    download_nc_files(
        years=selected_years, months=selected_months, data_dir=DATA_DIR
    )