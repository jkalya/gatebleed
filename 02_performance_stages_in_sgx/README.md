# performance_stages_in_sgx

This confirms that performance stages are still present even within an SGX enclave. 
This code performs the code in the performance_stages folder within an SGX enclave, printing the amount of time it took to execute an AMX operation
The presence of stages in the enclave means that all the attacks we describe in the paper are able to be run within SGX

Steps
- `make`
- `./sweep_waits.bash > data.csv`
- `./python3 plot.py`
- Plot is stored in plot.png

Troubleshooting
The Intel SGX SDK needs to be installed. On our system, it was installed to `/opt/intel/sgxsdk`, if it is different for you modify the `SGX_SDK` variable in `Makefile`
If you get an output of `SGX error code: 0x2006`, SGX needs to be enabled in the BIOS. We used [sgx-software-enable](https://github.com/intel/sgx-software-enable) to check if SGX was enabled

