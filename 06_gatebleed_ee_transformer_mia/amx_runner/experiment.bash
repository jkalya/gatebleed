#!/bin/bash

# First test set
TOTAL_LINES_TEST=$(wc -l < "../build/qa_test.txt")
# TOTAL_LINES_TEST=5

for i in $(seq 1 $TOTAL_LINES_TEST); do
	./launch_trial.bash $i "test"
	PID=$!
	wait $PID
	sleep 0.1
done

echo "Finished test"

# Then train set
TOTAL_LINES_TRAIN=$(wc -l < "../build/qa_train.txt")
# TOTAL_LINES_TRAIN=5


for i in $(seq 1 $TOTAL_LINES_TRAIN); do
	./launch_trial.bash $i "train"
	PID=$!
	wait $PID
	sleep 0.1
done

echo "Finished train"

# for i in {1..1000}; do
# 	./launch_trial.bash 1
# 	PID=$!
# 	wait $PID
# 	sleep 0.1
# done
