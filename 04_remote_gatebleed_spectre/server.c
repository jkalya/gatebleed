#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <stdint.h>

#include "../timer.h"
#include "network_config.h"
#include "../amx.h"

volatile uint64_t junk5;

__attribute__((aligned(4096))) uint64_t important_cache_line;
__attribute__((aligned(4096))) uint64_t random_variable_to_miss;

int __attribute__((aligned(4096))) byte;
int __attribute__((aligned(4096))) bit;

char secret[] = "It was the best of times, it was the worst of times";
char array_to_access[4] = {0, 0, 0, 0};
size_t __attribute__((aligned(4096))) array_to_access_size = 4*8;
//volatile char junk5;
//FILE* devnull;

int8_t src1[MAX];
int8_t src2[MAX];
int32_t res[MAX/4];

int __attribute__((aligned(4096))) byte;
int __attribute__((aligned(4096))) bit;

// Puts the cache line in the cache or not
void* amx_reset_command() {
	int listenfd = 0;
	struct sockaddr_in serv_addr;
	char buffer[BUFFER_SIZE]; // Don't even need to send anything

	listenfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	memset(&serv_addr, 0, sizeof(serv_addr));
	memset(buffer, 0, BUFFER_SIZE);

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(RESET_PORT);

	bind(listenfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

	socklen_t slen = sizeof(serv_addr);
	int received_val = 0;
	char recv_buffer[BUFFER_SIZE];

	while (1) {
		// Receive request
		recvfrom(listenfd, &recv_buffer, BUFFER_SIZE, 0, (struct sockaddr *) &serv_addr, &slen);
		//if (recv_buffer[0]) {
			//printf("Flush!\n");

			// Flush unconditionally
			//_mm_clflush(&important_cache_line);
		if (recv_buffer[0] == '1') {
			cooldown_amx_stage_3_tsc();
		}

			// Magic that just makes flush+reload work idk
			//for (volatile int z = 0; z < 200; z++) {}
			//}
		/* else { */
		/* 	//printf("NO flush!\n"); */
		/* 	//junk5 = important_cache_line; */
		/* 	_tile_dpbssd(1, 2, 3); */
		/* } */
		sendto(listenfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&serv_addr, slen);
	}
}

void* amx_leak_command() {
	//FILE* f;
	//f = fopen("/dev/null", "w");
	volatile uint64_t junk42;
	random_variable_to_miss = 42;
	int listenfd = 0;
	struct sockaddr_in serv_addr, cli_addr;
	char buffer[BUFFER_SIZE];
	//int received_val;
	char received_val[10];
	socklen_t slen = sizeof(serv_addr);

	listenfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	memset(&serv_addr, 0, sizeof(serv_addr));
	memset(buffer, 0, BUFFER_SIZE);

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(LEAK_PORT);

	bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

	while (1) {
		_mm_clflush(&random_variable_to_miss);
		_mm_mfence();
		// Receive request
		recvfrom(listenfd, &received_val, 10, 0, (struct sockaddr*)&cli_addr, &slen);
		//received_val = ntohl(received_val);

		//printf("In leak endpoint! I received: %d\n", offset);

		_mm_clflush(&array_to_access_size);
		
		// Magic that just makes flush+reload work idk
		/* for (volatile int z = 0; z < 200; z++) {} */
		/* for (volatile int z = 0; z < 200; z++) {} */
		/* for (volatile int z = 0; z < 200; z++) {} */
		/* for (volatile int z = 0; z < 200; z++) {} */
		/* for (volatile int z = 0; z < 200; z++) {} */


		
		
		int offset = atoi(received_val);
		
		// Access variable
		//TIMER_INIT();
		//_mm_mfence();
		//TIMER_START();
		//memcpy(buffer, &important_cache_line, 8);
		
		// For validating if AMX will execute or not execute based off of the contents of the message
		/* if (offset) { */
		/* 	fprintf(f, "asdf"); // If I uncomment this line, it magically works... idk */
		/* 	//junk42 = random_variable_to_miss; */
		/* 	_tile_dpbssd(1, 2, 3); */
		/* } */

		//_mm_mfence();
		//for (volatile int z = 0; z < 200; z++) {}
		//junk5 = important_cache_line;
		//TIMER_END();
		//_mm_mfence();

		for (volatile int z = 0; z < 200; z++) {} // this is required
		byte = offset/8;
		bit = abs(offset%8);
		//printf("%d\n", (array_to_access[byte] >> bit) & 0x01);
		if (offset < array_to_access_size) {
			if ((array_to_access[byte] >> bit) & 0x01) {
				//printf("%d\n", (array_to_access[byte] >> bit) & 0x01);
				_tile_dpbssd(1, 2, 3);
				_tile_dpbssd(1, 2, 3);
				/* _tile_dpbssd(1, 2, 3); */
				/* _tile_dpbssd(1, 2, 3); */
				/* _tile_dpbssd(1, 2, 3); */
				/* _tile_dpbssd(1, 2, 3); */
				/* _tile_dpbssd(1, 2, 3); */
				/* _tile_dpbssd(1, 2, 3); */
				/* _tile_dpbssd(1, 2, 3); */
				/* _tile_dpbssd(1, 2, 3); */
				/* _tile_dpbssd(1, 2, 3); */
			}
		}

		//printf("%"PRIu64"\n", TIMER_VALUE());
		sendto(listenfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&cli_addr, slen);
	}	
}

// Reads from the variable that is either cached or uncached, responds
void* cache_recover_command() {
	int listenfd = 0;
	struct sockaddr_in serv_addr, cli_addr;
	char buffer[BUFFER_SIZE];
	int received_val;
	socklen_t slen = sizeof(serv_addr);

	listenfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	memset(&serv_addr, 0, sizeof(serv_addr));
	memset(buffer, 0, BUFFER_SIZE);

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(RECOVER_PORT);

	bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

	WARMUP();
	
	while (1) {
		// Receive request
		recvfrom(listenfd, &received_val, sizeof(received_val), 0, (struct sockaddr*)&cli_addr, &slen);
		received_val = ntohl(received_val);

		// Access variable
		//TIMER_INIT();
		//_mm_mfence();
		//TIMER_START();
		//memcpy(buffer, &important_cache_line, 8);
		_tile_dpbssd(1, 2, 3);
		//junk5 = important_cache_line;
		//TIMER_END();
		//_mm_mfence();

		//printf("%"PRIu64"\n", TIMER_VALUE());
		sendto(listenfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&cli_addr, slen);
	}
}

int main() {
	/* // Initialize data for cache line of interest, ensure mapping in TLB */
	/* important_cache_line = 42; */
	/* junk5 = important_cache_line; */
	/* _mm_mfence(); */
	// Initialize AMX
	init_amx_all_tiles_max();

	int difference = (int)(array_to_access-secret);
	difference *= 8;
	printf("Starting point: %d\n", -difference-7);
	printf("Secret: %s\n", &array_to_access[-difference/8]);
	
	pthread_t reset_thread, recover_thread, leak_thread;
	int t;
	
    t = pthread_create(&reset_thread, NULL, amx_reset_command, NULL);
	t = pthread_create(&recover_thread, NULL, cache_recover_command, NULL);
	t = pthread_create(&leak_thread, NULL, amx_leak_command, NULL);
	
	pthread_join(reset_thread, NULL);
	pthread_join(recover_thread, NULL);
	pthread_join(leak_thread, NULL);
}

