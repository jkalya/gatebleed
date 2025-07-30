import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('amx_sniff_data.csv', header=None, names=['label', 'value'])
# df['value'] = df['value'].str.rstrip('%').astype(float)

# Separate data based on label (0 or 1)
values_0 = df[df['label'] == 0]['value']
values_1 = df[df['label'] == 1]['value']

# Plot histograms

plt.figure(figsize=(10, 6))
plt.hist(values_0, bins=30, alpha=0.6, label='Training Set', color='blue', density=True)
plt.hist(values_1, bins=30, alpha=0.6, label='Test Set', color='orange', density=True)

# Labels and legend
plt.title('Full MIA Attack')
plt.xlabel('% of fast AMX operations during inference')
plt.ylabel('Frequency')
plt.legend()
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
# plt.show()

# Save the plot as an image
plt.savefig('./histogram.png', format='png')
