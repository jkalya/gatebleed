import subprocess
import time
from datetime import datetime
from tqdm import tqdm
import os

# Use this when you want data in 1 file
# This is best when each run of the program outputs a single value
def experiment_runner(exe: str, nwarmups: int, ntrials: int, sleep_after_trial: float, suffix: str):
    
    # Warmup
    for i in tqdm(range(0, nwarmups)):
        subprocess.run(exe.split(), capture_output=True)
        time.sleep(sleep_after_trial);

    now = datetime.now()
    nowstr = now.strftime("%Y%m%d%H:%M:%S")
    filename = nowstr + "_" + suffix + ".csv"
        
    with open(filename, "a") as f:            
        for i in tqdm(range(0, ntrials)):
            out = subprocess.run(exe.split(), capture_output=True)
            out = out.stdout.decode().rstrip()
            print(out, file=f)
            time.sleep(sleep_after_trial)

# Don't use this - never finished but not sure if anything depends on it. Will eliminate when I clean this repo
def experiment_runner_2(exes, labels, nwarmups: int, ntrials: int, sleep_after_trial: float, folder: str, suffix: str):
    if not os.path.isdir(folder):
        os.makedirs(folder) # make the directory

    now = datetime.now()
    nowstr = now.strftime("%Y%m%d%H:%M:%S")
    filename = folder + "/" + nowstr + "_" + suffix + ".csv"

    assert len(exes) == len(labels)

    with open(filename, "a") as f:
        for e,l in zip(exes, labels):
            for i in tqdm(range(0, ntrials)):
                out = subprocess.run(e.split(), capture_output=True)
                out = out.stdout.decode().rstrip()
                print(str(l) + "," + out, file=f)
                time.sleep(sleep_after_trial)

# Use this when you want data in multiple files, all within a timestamped folder
# This is best when each run of the program outputs multiple values (e.g. curves)
def experiment_runner_3(exe: str, nwarmups: int, ntrials: int, sleep_after_trial: float, folder: str):
    now = datetime.now()
    nowstr = now.strftime("%Y%m%d%H:%M:%S")

    foldername = nowstr + folder

    os.makedirs(foldername)

    for i in tqdm(range(0, nwarmups)):
        subprocess.run(exe.split(), capture_output=True)
        time.sleep(sleep_after_trial)

    for i in tqdm(range(0, ntrials)):
        filename = foldername + "/" + "{:07d}".format(i) + ".csv"
        with open(filename, "a") as f:
            out = subprocess.run(exe.split(), capture_output=True)
            out = out.stdout.decode().rstrip()
            print(out, file=f)
            time.sleep(sleep_after_trial)

# Use these when sweeping values - instead of spamming the folder with a bunch of csvs, put a label with each point
# and put in same file
# First, capture the new filename by calling experiment_runner_new_file: this creates a new empty file
# Then, repeatedly call experiment_runner_sweep with the desired label and filename
def experiment_runner_new_file(suffix: str):
    now = datetime.now()
    nowstr = now.strftime("%Y%m%d%H:%M:%S")
    filename = nowstr + "_" + suffix + ".csv"
    with open(filename, 'w') as file:
        pass

    return filename

def experiment_runner_sweep(exe: str, nwarmups: int, ntrials: int, sleep_after_trial: float, label: str, filename: str):

    # Warmup
    for i in tqdm(range(0, nwarmups)):
        subprocess.run(exe.split(), capture_output=True)
        time.sleep(sleep_after_trial);

    with open(filename, "a") as f:
        for i in tqdm(range(0, ntrials)):
            out = subprocess.run(exe.split(), capture_output=True)
            out = out.stdout.decode().rstrip()
            print(label + "," + out, file=f)
            time.sleep(sleep_after_trial)

# Use when sweeping 2 aspects
def experiment_runner_sweep2(exe: str, nwarmups: int, ntrials: int, sleep_after_trial: float, label: str, label2: str, filename: str):

    # Warmup
    for i in tqdm(range(0, nwarmups)):
        subprocess.run(exe.split(), capture_output=True)
        time.sleep(sleep_after_trial);

    with open(filename, "a") as f:
        for i in tqdm(range(0, ntrials)):
            out = subprocess.run(exe.split(), capture_output=True)
            out = out.stdout.decode().rstrip()
            print(label + "," + label2 + "," + out, file=f)
            time.sleep(sleep_after_trial)

# Use when not sweeping at all - you just want multiple runs in the same file, the executables will output labels
def experiment_runner_sweep0(exe: str, nwarmups: int, ntrials: int, sleep_after_trial: float, filename: str):
    
    # Warmup
    for i in tqdm(range(0, nwarmups)):
        subprocess.run(exe.split(), capture_output=True)
        time.sleep(sleep_after_trial);

    with open(filename, "a") as f:
        for i in tqdm(range(0, ntrials)):
            out = subprocess.run(exe.split(), capture_output=True)
            out = out.stdout.decode().rstrip()
            print(out, file=f)
            time.sleep(sleep_after_trial)

# Use when sweeping, provide all arguments as list, multiple runs in same file, executable outputs labels, progress bar actually means something
def experiment_runner_usersweep0(exenoarg: str, args: list, nwarmups: int, ntrials: int, sleep_after_trial: float, filename: str):
    warmup_arg = args[0]
    # Warmup
    for i in tqdm(range(0, nwarmups)):
        split_command = (exenoarg + " " + str(warmup_arg)).split()
        subprocess.run(split_command, capture_output=True)
        time.sleep(sleep_after_trial);

    with open(filename, "a") as f:
        for a in tqdm(args):
            for i in range(0, ntrials):
                split_command = (exenoarg + " " + str(a)).split()
                out = subprocess.run(split_command, capture_output=True)
                out = out.stdout.decode().rstrip()
                print(out, file=f)
                time.sleep(sleep_after_trial)
    
