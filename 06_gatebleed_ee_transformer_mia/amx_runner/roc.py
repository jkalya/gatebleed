import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
df = pd.read_csv('amx_sniff_data.csv', header=None, names=['train', 'fast amx slices'])
fpr, tpr, thresholds = roc_curve(df["train"], df["fast amx slices"]);
plt.plot(fpr, tpr, lw=2, label=f'ROC')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')  # Diagonal line

plt.savefig('./roc.png', format='png')
