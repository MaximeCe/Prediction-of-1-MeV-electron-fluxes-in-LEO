"""file to load the data from the data folder."""
import os

DATA_DIR = os.getenv("DATA_DIR", "data")  # Default to "data" if not set


def iter_data(data_folder=DATA_DIR):
    """Yield one DataFrame per .nc file to keep memory use bounded."""
    import pandas as pd
    import xarray as xr

    nc_files = sorted(
        f for f in os.listdir(data_folder) if f.endswith(".nc")
    )
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {data_folder!r}")

    for nc_file in nc_files:
        file_path = os.path.join(data_folder, nc_file)
        with xr.open_dataset(file_path) as ds:
            yield pd.DataFrame(
                {
                    name: coord.values
                    for name, coord in ds.coords.items()
                    if coord.ndim == 1
                }
            )


def load_data(data_folder=DATA_DIR):
    """Load and concatenate every .nc file; use iter_data to limit memory use."""
    import pandas as pd

    return pd.concat(iter_data(data_folder), ignore_index=True)

if __name__ == "__main__":
    for index, chunk in enumerate(iter_data(), start=1):
        print(f"Loaded file {index}: {len(chunk)} rows")
        if index == 1:
            print(chunk.head())