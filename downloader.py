import os
import shutil
from typing import List
import requests

DATA_DIR = os.getenv("DATA_DIR", "data")

def download_nc_files(
    years: List[int], months: List[int], data_dir: str = DATA_DIR
) -> None:
    """Empties the target directory and downloads NOAA POES NetCDF files.

    Args:
        years: List of years (e.g., [2020, 2021]).
        months: List of months as integers (e.g., [1, 2, 12]).
        data_dir: Path to the local directory where files will be stored.
    """
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
        "access/l1b/v01r00/{year}/noaa15/poes_n15_{year}01{month_str}_proc.nc"
    )

    # 2. Download files based on provided years and months
    for year in years:
        for month in months:
            # Format month with leading zero (e.g., 1 -> '01')
            month_str = f"{month:02d}"
            file_url = base_url.format(year=year, month_str=month_str)
            filename = f"poes_n15_{year}01{month_str}_proc.nc"
            destination_path = os.path.join(data_dir, filename)

            print(f"Downloading: {file_url}")
            try:
                response = requests.get(file_url, stream=True, timeout=30)
                response.raise_for_status()

                with open(destination_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Successfully saved to {destination_path}")

            except requests.exceptions.RequestException as e:
                print(f"Failed to download {file_url}: {e}")


if __name__ == "__main__":
    # Example usage:
    selected_years = [2020]
    selected_months = list(range(1, 13))  # Months from 1 to 12

    download_nc_files(
        years=selected_years, months=selected_months, data_dir=DATA_DIR
    )