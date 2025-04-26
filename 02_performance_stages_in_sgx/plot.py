import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
#df = pd.read_csv("20250320_23:49.csv")
df = pd.read_csv("data.csv")

# Assuming the columns are unnamed, or named generically like 'A', 'B':
# If you know the column names, use them directly instead of df.columns[0] and df.columns[1]
grouped = df.groupby(df.columns[0])[df.columns[1]].median()

# Plotting
grouped.plot(kind='line')
plt.xscale("log")
plt.grid(True, which='both', linestyle='--')
#plt.show()
plt.savefig("plot.png", dpi=300, bbox_inches='tight')
