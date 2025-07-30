#!/bin/bash

cd build
echo -e "cold\n" | ./transformer_app
mv readings.txt readings_cold.txt
echo -e "warm\n" | ./transformer_app
mv readings.txt readings_warm.txt

cd ..

python3 plot.py
