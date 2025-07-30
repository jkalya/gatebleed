#!/bin/bash

# CORE=28

# # Get cmd line argument - whether to early exit or not
# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <argument>"
#     exit 1
# fi

# EARLY_EXIT=$1

# # Start AMX sniffer
# echo "$EARLY_EXIT,$(taskset -c $CORE ./main)" >> amx_sniff_data.csv &
# # taskset -c $CORE ./main > amx_sniff_data.csv &
# cd ../build && taskset -c $CORE ./infer $EARLY_EXIT 0 &

CORE=28



# Get cmd line argument - whether to early exit or not
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <line_number> <train_or_test>"
    exit 1
fi

PROMPT_FILE="../build/qa_train.txt"
if [[ "$2" == "test" ]]; then
	PROMPT_FILE="../build/qa_test.txt"
else
	PROMPT_FILE="../build/qa_train.txt"
fi


# Prepare the prompt file
# awk "NR==$1" ../build/qa_test.txt > ../build/the_prompt.txt
awk "NR==$1" $PROMPT_FILE > ../build/the_prompt.txt


# Run AMX sniffer in the background and log to a temporary file
taskset -c $CORE ./main > temp_amx_output.log 2>&1 &
MAIN_PID=$!

# Run inference (also in background if you want parallelism)
cd ../build && taskset -c $CORE ./infer &
INFER_PID=$!

# Wait for both to finish
wait $MAIN_PID
wait $INFER_PID

# Prefix main output with EARLY_EXIT and write to final CSV

if [ "$2" = "test" ]; then
	prefix=0
else
	prefix=1
fi
awk -v prefix="$prefix" '{ print prefix "," $0 }' temp_amx_output.log >> amx_sniff_data.csv

# awk '{ prefix = ($2 == "test") ? 0 : 1; print prefix "," $0 }' temp_amx_output.log >> amx_sniff_data.csv

# Optionally clean up temp file
rm temp_amx_output.log

echo "Both processes completed cleanly."



