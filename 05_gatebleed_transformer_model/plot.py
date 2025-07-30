import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize an empty list to store the first column values
first_column_data_cold = []
first_column_data_warm = []

# Remove outliers and smooth the data
def clean_and_smooth(arr):
    median_val = np.median(arr)
    mad = np.median(np.abs(arr - median_val))  # Median absolute deviation
    # Identify outliers
    deviation = np.abs(arr - median_val)
    arr_no_outliers = arr[deviation <= 3 * mad] # Remove anything outside 3 MAD
    if len(arr_no_outliers) >= 32:
        return pd.Series(arr_no_outliers).rolling(window=2, min_periods=1).mean().values # Mean filter
    else:
        return arr_no_outliers  # not enough data to smooth

# Read the file
with open('./build/readings_cold.txt', 'r') as file:
    for line in file:
        # Skip empty lines
        if line.strip():
            # Split the line by whitespace and convert the first column to an integer
            first_column_data_cold.append(int(line.split()[0]))

with open('./build/readings_warm.txt', 'r') as file:
    for line in file:
        # Skip empty lines
        if line.strip():
            # Split the line by whitespace and convert the first column to an integer
            first_column_data_warm.append(int(line.split()[0]))

first_column_data_cold = clean_and_smooth(np.array(first_column_data_cold))
first_column_data_warm = clean_and_smooth(np.array(first_column_data_warm))

# Plot the first column data as a histogram
plt.hist(first_column_data_cold, bins=30, edgecolor='blue')
plt.hist(first_column_data_warm, bins=30, edgecolor='red')

# Set the labels
plt.xlabel('time(cycle)')
plt.ylabel('frequency')

# Save the plot as an image
plt.savefig('./build/histogram.png', format='png')
