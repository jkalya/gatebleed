import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

plt.plot(df.iloc[:, 1], df.iloc[:, 2])
plt.xscale('log')
plt.grid(True)

#plt.show()
plt.savefig('plot.png')
