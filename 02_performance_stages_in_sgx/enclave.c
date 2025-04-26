#include <stdint.h>
//#include <sgx_intrin.h>

#include "enclave_t.h"

typedef struct __tile_config
{
	uint8_t palette_id;
	uint8_t start_row;
	uint8_t reserved_0[14];
	uint16_t colsb[8];
	uint8_t reserved_0_again[16];
	uint8_t rows[8];
	uint8_t reserved_0_again_again[8];
} __tilecfg;

uint64_t gadget(uint64_t wait) {
	int junk;
	// Initialize AMX (done every time because SGX is stateless)
	__tilecfg c = {0};
	c.palette_id = 1;
	c.start_row = 0;
	c.colsb[0] = 64;
	c.colsb[1] = 64;
	c.colsb[2] = 64;
	c.colsb[3] = 64;
	c.rows[0] = 16;
	c.rows[1] = 16;
	c.rows[2] = 16;
	c.rows[3] = 16;
	__asm__ volatile ("ldtilecfg\t%X0" :: "m" (*((const void**)&c))); // from amxtileintrin.h

	asm volatile("TDPBSSD %tmm0, %tmm1, %tmm2\n");

	volatile uint64_t the_wait = wait;

	while (the_wait != 0) {
		the_wait--;
	}

	uint64_t start;
	uint64_t end;

	/* asm volatile("RDTSCP\n" */
	/* 			 "MOVL %%ecx, %%eax\n" */
	/* 			 "SHLQ $32, %%rcx\n" */
	/* 			 "MOVL %%ecx, %%edx\n" */
	/* 			 "MOVQ %0, %%rcx\n" */
	/* 			 : "=r" (start) */
	/* 			 : */
	/* 			 :"rcx", "eax", "edx"); */

	asm volatile ( "rdtscp\n\t"    // Returns the time in EDX:EAX.
				   "shl $32, %%rdx\n\t"  // Shift the upper bits left.
				   "or %%rdx, %0"        // 'Or' in the lower bits.
				   : "=a" (start)
				   : 
				   : "rdx");
	
	asm volatile("TDPBSSD %tmm0, %tmm1, %tmm2\n");
	
	asm volatile ( "rdtscp\n\t"    // Returns the time in EDX:EAX.
				   "shl $32, %%rdx\n\t"  // Shift the upper bits left.
				   "or %%rdx, %0"        // 'Or' in the lower bits.
				   : "=a" (end)
				   : 
				   : "rdx");
	
	//uint64_t end = _rdtscp(&junk);

	return end-start;
	
}
