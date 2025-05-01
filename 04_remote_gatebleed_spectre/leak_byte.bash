#!/bin/bash

# Check if an argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <offset>"
  exit 1
fi

# Get the input number
start_num=$1

for i in $(seq $start_num $((start_num + 7))); do
  ./client $i >> "data_$start_num.csv"
done
