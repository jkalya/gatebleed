#include <stdio.h>
#include "../amx.h"
#include "../timer.h"

volatile uint64_t decrement = 0;

int main(int argc, char* argv[]) {
	if (argc != 3) {
		printf("Syntax: ./busyloop WARMUP WAIT_TIME\n");
		return 1;
	}

	init_amx_all_tiles_max();

	decrement = atoll(argv[2]);

	// only needed if c1e is enabled
	if (atoi(argv[1])) {
		WARMUP();
	}

	int junk;
	uint64_t target = _rdtscp(&junk) + decrement;

	asm volatile("TDPBSSD %tmm0, %tmm1, %tmm2\n");


	while (_rdtscp(&junk) < target) {}

	TIMER_START();
	asm volatile("TDPBSSD %tmm3, %tmm4, %tmm5\n");
	TIMER_END();

	printf("%"PRIu64",%d,%"PRIu64"\n", atoll(argv[1]), atoi(argv[2]), TIMER_VALUE());

	return 0;
}
