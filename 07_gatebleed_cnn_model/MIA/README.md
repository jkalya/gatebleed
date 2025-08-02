#  Neural Network in Pure C++

Simple modular implementation of a neural network in C++ using only the STL, with support for convolutional layers, ReLU, MaxPooling, early exit, and AMX profiling.

Info to remember - Entropy threshold is set to 2.15266

---

### 1.  Configure and Build

```bash
make
```

---

##  Run Inference and Generate Entropy Graph

Run the following Python script to:
- Build the project
- Execute inference for both member and non-member samples
- Save inference timing and entropy results
- Plot entropy histogram

```bash
python3 python_run_and_plot_graph.py
```

This will:
- Use `taskset` to bind execution to a core
- Write output to:
  - `test_non-member.txt`
  - `train_member.txt`
- Generate AMX timing histogram to infer Membership (`MIA_Gatebleed_CNN_graph.png`)

---

##  Manual Inference (Optional)

```bash
# For non-members
sudo taskset -c 28 ./build/neural_net_in_cpp data test1 0 0.01 5000 test_1_non-member.txt > entropy_non-member_1.txt

# For members
sudo taskset -c 28 ./build/neural_net_in_cpp data train3 1 0.01 5000 train_3_member.txt > entropy_member_3.txt
```

### Command Format

```bash
./neural_net_in_cpp <data_path> <loader_type> <membership_label> <threshold> <extra_ops> <output_file>
```

- `data_path`: Path to MNIST dataset
- `loader_type`: `test`, `train`, etc.
- `membership_label`: `0` = non-member, `1` = member
- `threshold`: Confidence threshold (e.g., 0.01)
- `extra_ops`: Dummy operations for timing balance
- `output_file`: File where AMX timing and entropy logs are saved

---

## Features

- [x] Fully Connected Layers  
- [x] ReLU Activation  
- [x] Softmax Classifier  
- [x] Convolutional Layers  
- [x] MaxPooling  
- [x] Learning Rate Scheduler  
- [x] Early Exit  
- [x] AMX Instruction Profiling  

---

## 📄 License

MIT
