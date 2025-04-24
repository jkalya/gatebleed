#!/bin/bash

waits=$(seq 1 10)
#waits+=$(seq 10 100 10)

for i in "$waits"
do
	echo "${i}_test"
done
