# gatebleed artifact

This artifact was tested on a Lenovo SRV-650 V3 with dual-socket Intel Xeon Gold 5420+ (Sapphire Rapids) CPUs, running either RHEL 9.4 or Ubuntu 22.04. The artifacts are as follows

## 01_performance_stages
Shows the different AMX instruction latencies depending on time AMX was last utilized.

## 02_performance_stages_in_sgx
Shows that these performance stages happen in Intel SGX (a trusted execution environment), showing that attacks that build off the performance stages also work when the application is enclosed in an SGX enclave.

## 03_remote_gatebleed_covert_channel
Shows a remote covert channel with cooperating sender and receiver.

## 04_remote_gatebleed_spectre
Shows a PoC remote spectre attack using the gatebleed covert channel. 

## 05_gatebleed_transformer_model
Shows how execution time of a realistic transformer model is severely affected by whether AMX was warmed up or not. 

# Troubleshooting
More specific troubleshooting is provided in the individual artifacts, but here are some overall tips

- Our server was a Lenovo SR650 V3 with Sapphire Rapids CPUs
- Our attacks worked on RHEL 9.4 and Ubuntu 22.04
- The covert channel and side-channel attack (03 and 04) works best when C-states are disabled and C1e is disabled
- If you can't disable C-states or C1e but SMT (Hyperthreading) is enabled, use the busyloop programs provided with 03 and 04
- To use the busyloop program, pin the busyloop program to the sibling hyperthread
For example, if we ran "server" with `taskset -c 0 ./server` and our server is a dual-socket 56-core setup for a total of 112 cores, run `taskset -c 56 ./busyloop` 
Sibling hyperthreads can be found by invoking `cat /sys/devices/system/cpu/cpuN/topology/thread_siblings_list`, replacing N in cpuN with the desired core
- Programs work best when pinned to a core, e.g. `taskset -c 0 PROGRAM`
