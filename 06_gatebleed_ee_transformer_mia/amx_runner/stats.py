import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix


df = pd.read_csv('amx_sniff_data.csv', header=None, names=['train', 'fast amx slices'])

fpr, tpr, thresholds = roc_curve(df["train"], df["fast amx slices"]);

y_true = df["train"]
y_scores = df["fast amx slices"]

# Find the best threshold (Youden's J statistic)
j_scores = tpr - fpr
best_index = j_scores.argmax()
best_threshold = thresholds[best_index]

y_pred = (y_scores >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0.0
accuracy = accuracy_score(y_true, y_pred)


# Compute rates
total_positives = tp + fn
total_negatives = tn + fp

tpr_value = tp / total_positives if total_positives > 0 else 0.0
fnr_value = fn / total_positives if total_positives > 0 else 0.0
tnr_value = tn / total_negatives if total_negatives > 0 else 0.0
fpr_value = fp / total_negatives if total_negatives > 0 else 0.0

print(f"Best Threshold: {best_threshold:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")
print(f"True Positive Rate (TPR / Sensitivity): {tpr_value:.4f}")
print(f"False Negative Rate (FNR): {fnr_value:.4f}")
print(f"True Negative Rate (TNR / Specificity): {tnr_value:.4f}")
print(f"False Positive Rate (FPR): {fpr_value:.4f}")
print(f"Precision: {tp/(tp+fp)}")
