import sys
sys.path.append("../../")

from experiment_runner import *

#experiment_runner_sweep("taskset -c 29 ./client 0", 100, 10000, 0.0, "0", f)
#experiment_runner_sweep("taskset -c 29 ./client 1", 100, 10000, 0.0, "1", f)

experiment_runner("taskset -c 28 ./main 0", 0, 100, 0.0, "avx_contends_amx")
