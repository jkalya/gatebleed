#!/bin/bash

waits=({1..9})
waits+=({10..100..10})
waits+=({100..10000..100})
waits+=({10000..100000..1000})
waits+=({100000..50000000..100000})
#waits+=({20000000..40000000..1000})

#echo $waits

for item in "${waits[@]}"; do
	for i in {1..10};
	do		 
		taskset -c 1 ./main $item
	done
done

