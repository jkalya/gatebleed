import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data.csv")

# Column names
group_col = df.columns[0]
value_col = df.columns[1]

# IQR-based outlier removal
def remove_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.0 * iqr
    upper = q3 + 1.0 * iqr
    return series[(series >= lower) & (series <= upper)]

# Get filtered values
data_0 = remove_outliers(df[df[group_col] == 0][value_col])
data_1 = remove_outliers(df[df[group_col] == 1][value_col])

# Define common bin edges
#bins = np.linspace(min(df[value_col]), max(df[value_col]), 1000)

# Compute histograms
counts_0, bin_edges = np.histogram(data_0, bins=100)
counts_1, bin_edges_1 = np.histogram(data_1, bins=100)

# Smooth using a moving average (window size = 3)
def smooth_counts(counts, window=3):
    return pd.Series(counts).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

smooth_0 = smooth_counts(counts_0, window=1)
smooth_1 = smooth_counts(counts_1, window=1)

# Midpoints for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_centers_1 = (bin_edges_1[:-1] + bin_edges[1:]) / 2

# Plotting
plt.plot(bin_centers, smooth_0, label='AMX hot', linewidth=2)
plt.plot(bin_centers_1, smooth_1, label='AMX cold', linewidth=2)

plt.ylabel("Latency")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("plot.png", dpi=300, bbox_inches='tight')
#plt.show()
