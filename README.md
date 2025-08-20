# gatebleed artifact

This artifact was tested on a Lenovo SRV-650 V3 with dual-socket Intel Xeon Gold 5420+ (Sapphire Rapids) CPUs, running both on RHEL 9.4 and Ubuntu 22.04. The artifacts are as follows

## 01_performance_stages
Shows the different AMX instruction latencies depending on time AMX was last utilized.

## 03_remote_gatebleed_covert_channel
Shows a remote covert channel with cooperating sender and receiver.

## 04_remote_gatebleed_spectre
Shows a PoC remote spectre attack using the gatebleed covert channel. 

## 05_gatebleed_transformer_model
Shows how execution time of a transformer model is severely affected by whether AMX was warmed up or not. The transformer model takes in a token sequence as input and outputs whether or not the input sequence was a question or answer. 

## 06_gatebleed_ee_transformer_mia
Shows a membership inference attack on a simplified entropy-based mixture-of-experts transformer model by showing how the attacker can measure its own AMX instruction execution time to determine if AMX was used in the previous time slice when the attacker can colocate its program on the same core as the victim running a machine learning inference. The number of attacker-observed fast AMX instructions will differ based on if the victim used a lower-capacity expert or not, and it is known that for overfitted models, intermediate entropy is different on training members vs nonmembers thus an expert decision based on entropy can leak to the attacker. The attacker is allowed to query the model, so the attacker can continually present inputs to the network in order to deduce what texts it was trained on. 

## 07_gatebleed_cnn_mia
Shows the member ship inference attack on simplified CNN model, here the attacker can measure the time taken by the member and non-members to get executed. If we are trying to infer a member, the entropy value might be low, which can cause the us the model to take early exit route. In order to compensate for the reduced time taken by the early exit, we add some extra computations so it seems like model has not taken the early exit route. But we can leak the member by measuring the AMX instruction after the inference of one image. AMX instruction after Member image will take more time to execute, because we extra computations are non-AMX operations, which cause AMX unit to powergate, whereas Non-Members will keep AMX unit in hot stage thus taking less time to execute the next AMX operation. 

# Troubleshooting
More specific troubleshooting is provided in the individual artifacts, but here are some overall tips

- Our server was a Lenovo SR650 V3 with Sapphire Rapids CPUs
- Our attacks worked on RHEL 9.4 and Ubuntu 22.04
- The best working Linux kernel version was 5.14.0-427.35.1.el9
- The covert channel and side-channel attack (03 and 04) works best when C-states are disabled and C1e is disabled
- If you can't disable C-states or C1e but SMT (Hyperthreading) is enabled, use the busyloop programs provided with 03 and 04
- To use the busyloop program, pin the busyloop program to the sibling hyperthread
For example, if we ran "server" with `taskset -c 0 ./server` and our server is a dual-socket 56-core setup for a total of 112 cores, run `taskset -c 56 ./busyloop` 
Sibling hyperthreads can be found by invoking `cat /sys/devices/system/cpu/cpuN/topology/thread_siblings_list`, replacing N in cpuN with the desired core
- Programs work best when pinned to a core, e.g. `taskset -c 0 PROGRAM`
