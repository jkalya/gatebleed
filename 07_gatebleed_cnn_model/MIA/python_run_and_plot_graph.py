import subprocess
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Step 1: Build the project ===
print("🔧 Building project...")
subprocess.run(["make"], check=True)

# === Step 2: Run inference and generate labeled output files ===
commands = [
    ["sudo", "taskset", "-c", "28", "./build/neural_net_in_cpp", "data", "test", "0", "0.01", "5000", "test_non-member.txt"],
    ["sudo", "taskset", "-c", "28", "./build/neural_net_in_cpp", "data", "train", "1", "0.01", "5000", "train_member.txt"]
]

output_files = ["test_non-member.txt", "train_member.txt"]

for cmd in commands:
    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# === Step 3: Parse entropy (1st col) and label (4th col) from files ===
def parse_entropy_and_labels(file_path):
    entropy = []
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    entropy_val = float(parts[0])
                    label = int(parts[3])
                    entropy.append(entropy_val)
                    labels.append(label)
                except:
                    continue
    return np.array(entropy), np.array(labels)

entropy_non_member, labels_non_member = parse_entropy_and_labels("test_non-member.txt")
entropy_member, labels_member = parse_entropy_and_labels("train_member.txt")

# === Step 4: Combine for prediction ===
threshold = 0.01  # Threshold for confidence

X = np.concatenate([entropy_member, entropy_non_member])
Y_true = np.concatenate([labels_member, labels_non_member])
Y_pred = (X < threshold).astype(int)

# === Step 1: Load AMX timing data and labels ===
def parse_amx_cycles_and_labels(filename):
    cycles = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    cycles.append(int(parts[0]))     # AMX cycle count
                    labels.append(int(parts[3]))     # Member (1) or Non-Member (0)
                except:
                    continue
    return np.array(cycles), np.array(labels)

# === Step 2: Remove outliers using IQR method ===
def remove_outliers(data, multiplier=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return data[(data >= lower) & (data <= upper)]

# Load both files
files = ["test_non-member.txt", "train_member.txt"]
all_cycles, all_labels = [], []

for file in files:
    cycles, labels = parse_amx_cycles_and_labels(file)
    all_cycles.extend(cycles)
    all_labels.extend(labels)

all_cycles = np.array(all_cycles)
all_labels = np.array(all_labels)

# === Step 3: Separate and filter member/non-member timings ===
member_cycles_raw = all_cycles[all_labels == 1]
non_member_cycles_raw = all_cycles[all_labels == 0]

member_cycles = remove_outliers(member_cycles_raw)
non_member_cycles = remove_outliers(non_member_cycles_raw)

# === Step 4: Compute means and threshold ===
mean_member = np.mean(member_cycles)
mean_non_member = np.mean(non_member_cycles)
threshold = int((mean_member + mean_non_member) / 2)

# === Step 5: Plot histogram ===
plt.figure(figsize=(10, 6))
plt.hist(non_member_cycles, bins=100, alpha=0.7, label="Non-Member", color="blue")
plt.hist(member_cycles, bins=100, alpha=0.7, label="Member", color="orange")

# Plot mean lines
plt.axvline(mean_non_member, color='green', linestyle='--', linewidth=2, label=f"Mean Non-Member: {mean_non_member:.2f}")
plt.axvline(mean_member, color='red', linestyle='--', linewidth=2, label=f"Mean Member: {mean_member:.2f}")
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f"Threshold: {threshold} cycles")

# Labels and title
plt.xlabel("Cycles")
plt.ylabel("Frequency")
plt.title("Time taken by AMX instruction after Member and Non Member")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show
plt.savefig("MIA_Gatebleed_CNN_graph.png")
plt.show()