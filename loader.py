"""file to load the data from the data folder."""
import os
DATA_DIR = os.getenv("DATA_DIR", "data")  # Default to "data" if not set

def load_data(data_folder=DATA_DIR):
    """Load all .nc files from the specified data folder into a single dataframe."""
    import os
    import pandas as pd
    import xarray as xr

    # List all .nc files in the data folder
    nc_files = [f for f in os.listdir(data_folder) if f.endswith('.nc')]

    # Load each .nc file into a DataFrame and concatenate them
    dataframes = []
    for nc_file in nc_files:
        file_path = os.path.join(data_folder, nc_file)
        ds = xr.open_dataset(file_path)
        df = ds.to_dataframe().reset_index()
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

if __name__ == "__main__":
    # Example usage
    combined_data = load_data()
    print(combined_data.head())