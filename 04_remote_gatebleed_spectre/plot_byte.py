import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import zscore
import sys

if len(sys.argv) != 3:
   print("Usage: python3 plot_byte.py <csv_file> <dividing line x>")
   sys.exit(1)

csv_file = sys.argv[1]
midpoint = int(sys.argv[2])

# Load CSV file (no header assumed, delimiter is comma)
data = pd.read_csv(csv_file, header=None)

# Get unique values from first column
unique_ids = data[0].unique()

# Group data by the first column (like cell array in MATLAB)
byte_data = [data[data[0] == uid][1].values for uid in unique_ids]

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

byte_data = [clean_and_smooth(b) for b in byte_data]

# Compute midpoint between mean of byte_data[1] and byte_data[3] (MATLAB is 1-indexed)
#midpoint = 19924 # set by command line argument now
# Plotting
#fig, axs = plt.subplots(8, 1, figsize=(16,15))
fig = plt.figure(figsize=(16,15))
gs = fig.add_gridspec(8, 1, hspace=0.4)
axs = [None] * 8

for i in range(8):
    axs[i] = fig.add_subplot(gs[i, 0])
    bdata = byte_data[i]
    x, bin_edges = np.histogram(bdata, bins='auto')
    n = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centers

    axs[i].plot(n, x, color='black', linewidth=2)

    y_min, y_max = axs[i].get_ylim()
    halfway = (y_min + y_max) / 2

    mean_val = np.mean(bdata)
    axs[i].plot(mean_val, halfway, 'o', color='blue', markersize=8)

    axs[i].axvline(midpoint, color='black', linewidth=2)

    axs[i].set_xlim([-20000, 60000])
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_box_aspect(1)  # force square subplot boxes

    label = "  1" if mean_val < midpoint else "  0"
    axs[i].text(mean_val, halfway, label, fontsize=20, color=(0.0, 0.5, 0.0))

    #axs[i].text(midpoint + 22000, halfway, "SUCCESS", fontsize=18, color=(0.0, 0.5, 0.0))

    axs[i].grid(True)
    axs[i].set_box_aspect(1)

plt.subplots_adjust(hspace=0.6, left=0.1, right=0.9)  # Adjust horizontal space and subplot size
#plt.tight_layout()
#plt.show()
plt.savefig(csv_file + ".png", dpi=300)
