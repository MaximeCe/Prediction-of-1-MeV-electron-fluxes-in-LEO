"""functions for visualizing the data."""


def print_column_names(data):
    """Print the column names of the data."""
    print("Column names:")
    for column in data.columns:
        print(column)

def plot_histogram(data, column_name, bins=10):
    """Plot a histogram of the specified column."""
    import matplotlib.pyplot as plt

    plt.hist(data[column_name].dropna(), bins=bins)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()
    
    
if __name__ == "__main__":
    # Example usage
    import pandas as pd 
    from loader import iter_data
    data = iter_data().__next__()  # Load the first chunk of data
    
    print_column_names(data)
    plot_histogram(data, 'mep_pro_tel0_flux_p1', bins=100)