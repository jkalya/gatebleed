#ifndef TIMER_H
#define TIMER_H

#include <x86intrin.h>
#include <time.h>
#include <inttypes.h>
//#include <papi.h>

#define TSC 0
//#define PAPI 1

#ifndef TIMER_TYPE
#define TIMER_TYPE TSC
#endif

// warm up cpu
uint32_t warmup_timer;
#define WARMUP() warmup_timer = 10000000; while (warmup_timer > 0) {warmup_timer--;}

uint64_t t_start, t_end;
unsigned int dump;

#if TIMER_TYPE==TSC

#warning "Using TSC timer"
#define TIMER_INIT() t_start = 0; t_end = 0;
#define TIMER_START() __asm__ __volatile__ ("LFENCE\n\t"); t_start = __rdtsc();
//#define TIMER_END() __asm__ __volatile__ ("MFENCE\n\t"); t_end = __rdtscp(&dump);
#define TIMER_END() t_end = __rdtscp(&dump);

//#elif TIMER_TYPE==PAPI

//#warning "Using PAPI timer"
//#define TIMER_INIT() PAPI_library_init(PAPI_VER_CURRENT); t_start = 0; t_end = 0;
//#define TIMER_START() t_start = PAPI_get_real_cyc();
//#define TIMER_END() __asm__ __volatile__ ("MFENCE\n\t"); t_end = __rdtscp(&dump);
//#define TIMER_END() t_end = PAPI_get_real_cyc();

#elif TIMER_TYPE==TSC2
#warning "Using TSC timer with only rdtscp"
#define TIMER_INIT() t_start = 0; t_end = 0;
#define TIMER_START() t_start = __rdtscp(&dump);
#define TIMER_END() t_end = __rdtscp(&dump);

#else

#error "Invalid timer, valid timers are TSC, PAPI, and TSC2"

#endif

#define TIMER_VALUE() t_end-t_start
#define TOUCH_BYTE(x) __asm__ __volatile__ ("movb %0, %%al;" \
											:"=r"(x)		 \
											:				 \
											:"%al");
#endif
