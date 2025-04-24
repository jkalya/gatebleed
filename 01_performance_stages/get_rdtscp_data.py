import sys
sys.path.append("../")
from experiment_runner import *

# 0-5 ns in increments of 1 ns (9)
waits = list(range(0,10))
# 5-50 ns in increments of 5 ns (9)
waits += list(range(10, 100, 10))
# 50 ns-50000 ns in increments of 50 ns (499)
waits += list(range(100, 100000, 100))
# 100 us - 30 ms in increments of 100 us (299)
waits += list(range(200000, 60000000, 200000))

# Grand total: 425 things

f = experiment_runner_new_file("minpow_nowarmup_rdtsc")
experiment_runner_usersweep0("taskset -c 0 ./busyloop 1 ", waits, 100, 1, 0.001, f)
