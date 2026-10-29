"""Pre-processing pipeline for the data."""

from loader import load_data

# FLAG
DOWNLOAD = False  # Set to True to download the data, False to use existing data
VISUALIZE = False  # Set to True to visualize the data, False to skip visualization
CLEAN = False  # Set to True to clean the data, False to skip cleaning 