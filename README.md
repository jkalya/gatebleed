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
