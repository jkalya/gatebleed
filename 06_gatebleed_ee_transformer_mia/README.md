# Transformer Inference Execution Time

This repository contains an MIA on an entropy-based early exit transformer model. The attack code is in amx_sniffer

Steps
- make all
- cd amx_sniffer
- make
- ./experiment.bash
- python3 plot.py
- python3 roc.py
- python3 stats.py

histogram.png should look like histogram_original.png, roc.png should look like roc_original.png, and stats.py should output something like this

```
Best Threshold: 98.0000
Accuracy: 0.8066
True Positives: 381
False Positives: 47
True Negatives: 253
False Negatives: 105
True Positive Rate (TPR / Sensitivity): 0.7840
False Negative Rate (FNR): 0.2160
True Negative Rate (TNR / Specificity): 0.8433
False Positive Rate (FPR): 0.1567
Precision: 0.8901869158878505
```
