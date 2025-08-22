#include <inttypes.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <assert.h>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

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

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
   {
	   //printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else
   {
	   //printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }

   return true;
}

bool init_amx(uint8_t* row_lengths, uint16_t* col_lengths, int num_lengths) {

	/* for (int i = 0; i < num_lengths; i++) { */
	/* 	printf("%d\t", row_lengths[i]); */
	/* } */
	/* printf("\n"); */
	/* for (int i = 0; i < num_lengths; i++) { */
	/* 	printf("%d\t", col_lengths[i]); */
	/* } */
	/* printf("\n"); */

			

	if (!set_tiledata_use()) {
		printf("Failed to do XFEATURE_XTILEDATA\n");
		return false;
	}
	
	__tilecfg c = {0};
	c.palette_id = 1;
	c.start_row = 0;

	//c.colsb[0] = MAX_ROWS;
	//c.rows[0] = MAX_ROWS;

	for (int i = 0; i < num_lengths; i++) {
		c.colsb[i] = col_lengths[i];
		c.rows[i] = row_lengths[i];
		//c.rows[i+1] = MAX_ROWS;
		//c.colsb[i+1] = MAX_COLS;
		//c.rows[i+1] = 16;
		//c.colsb[i+1] = 16;
	}

	_tile_loadconfig(&c);
	return true;
}

bool init_amx_all_tiles_max() {
	uint8_t row_lengths[] = {MAX_ROWS, MAX_ROWS, MAX_ROWS, MAX_ROWS, MAX_ROWS, MAX_ROWS, MAX_ROWS, MAX_ROWS};
	uint16_t col_lengths[] = {MAX_COLS, MAX_COLS, MAX_COLS, MAX_COLS, MAX_COLS, MAX_COLS, MAX_COLS, MAX_COLS};
	return init_amx(row_lengths, col_lengths, 8);
}

/* Print int8_t buffer */
static void print_buffer(int8_t* buf, int32_t rows, int32_t colsb) 
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%d ", buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}

/* Print int32_t buffer */
static void print_buffer32(int32_t* buf, int32_t rows, int32_t colsb)
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%d ", buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}

/* Initialize int8_t buffer */
static void init_buffer (int8_t *buf, int8_t value)
{
  int rows, colsb, i, j;
  rows  = MAX_ROWS;
  colsb = MAX_COLS;

  for (i = 0; i < rows; i++)
    for (j = 0; j < colsb; j++)
    {
        buf[i * colsb + j] = value;
    }
}

/* Initialize uint8_t buffer */
static void init_buffer_uint8 (uint8_t *buf, uint8_t value)
{
  int rows, colsb, i, j;
  rows  = MAX_ROWS;
  colsb = MAX_COLS;

  for (i = 0; i < rows; i++)
    for (j = 0; j < colsb; j++)
    {
        buf[i * colsb + j] = value;
    }
}

static void init_buffer_eye(size_t rows, size_t cols, int8_t* buf) {
	assert(rows == cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (i == j) {
				buf[i*cols+j] = 1;
			}
			else {
				buf[i*cols+j] = 0;
			}
		}
	}
}

/* Initialize int32_t buffer */
static void init_buffer32 (int32_t *buf, int32_t value)
{
  int rows, colsb, i, j;
  rows  = MAX_ROWS;
  colsb = MAX_COLS;
  int colsb2=colsb/4;

  for (i = 0; i < rows; i++) {
	  for (j = 0; j < (colsb2); j++) {
		  buf[i * colsb2 + j] = value;
	  }
  }
}

static void init_buffer32_size (size_t rows, size_t cols, int32_t *buf, int32_t value)
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			buf[i * cols + j] = value;
		}
	}
}

static void cooldown_amx_stage_1() {
	int i = 300;
	while (i >= 0) {
		i--;
	}
}

static void cooldown_amx_stage_2() {
	int i = 10000;
	while (i >= 0) {
		i--;
	}
}

static void cooldown_amx_stage_3() {
	int i = 25005000;
	while (i >= 0) {
		i--;
	}
}

static void cooldown_amx_stage_4() {
	int i = 50000000;
	while (i >= 0) {
		i--;
	}
}

static inline void cooldown_amx_stage_1_tsc() {
	int junk;
	uint64_t the_start = _rdtscp(&junk);
	while (_rdtscp(&junk) < the_start+1250) {}
}

static inline void cooldown_amx_stage_2_tsc() {
	int junk;
	uint64_t the_start = _rdtscp(&junk);
	while (_rdtscp(&junk) < the_start+10000) {}
}

static inline void cooldown_amx_stage_3_tsc() {
	int junk;
	uint64_t the_start = _rdtscp(&junk);
	while (_rdtscp(&junk) < the_start+50000000) {}
}

static inline void cooldown_amx_stage_4_tsc() {
	int junk;
	uint64_t the_start = _rdtscp(&junk);
	while (_rdtscp(&junk) < the_start+50000000) {}
}

// Causes a tile load with n page faults
// Up to caller to ensure base_addr buffer is large enough
static inline void n_pf_tileload(uint8_t* base_addr, int n, unsigned short tile_num) {
	_tile_loadd(4, base_addr, (n*4096)/16);
}
