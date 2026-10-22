"""Cleaner functions for cleaning up the data."""

def clean_data(data, saturated_value=9999):
    """Clean the data by replacing empty and saturated values with NaN."""
    data = data.where(data != 0, float('nan'))  # Replace empty values (0) with NaN
    data = data.where(data != saturated_value, float('nan'))  # Replace saturated
    return data