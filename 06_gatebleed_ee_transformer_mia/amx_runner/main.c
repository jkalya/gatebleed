#include <stdio.h>
#include "timer.h"
#include "amx.h"

//#define NTRIALS 15000
#define NTRIALS 1500
#define THRESHOLD 1000
uint64_t times[NTRIALS];

int main() {
	TIMER_INIT();
	if (!init_amx_all_tiles_max()) {
		return 1;
	}

	for (int i = 0; i < NTRIALS; i++) {
		TIMER_START();
		_tile_dpbssd(1, 2, 3);
		TIMER_END();
		//printf("%"PRIu64"\n", TIMER_VALUE());
		times[i] = TIMER_VALUE();
		usleep(1000);
		//cooldown_amx_stage_3();
	}

	int nwarm = 0;
	for (int i = 0; i < NTRIALS; i++) {
		nwarm += (times[i] < THRESHOLD);
	}

	//printf("%lf\%\n", ((double)nwarm/(double)NTRIALS)*100.0);
	printf("%d\n", nwarm);
}
