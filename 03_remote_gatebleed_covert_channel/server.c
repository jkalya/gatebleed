
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>

#include "network_config.h" // Change this when copying
#include "../timer.h"
#include "../amx.h"

void* reset_thread();
void* leak_thread();
void* recover_thread();

void* reset_function() {
	int listenfd = 0;
	struct sockaddr_in serv_addr;

	listenfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	memset(&serv_addr, 0, sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(RESET_PORT);

	bind(listenfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

	socklen_t slen = sizeof(serv_addr);
	char recv_buffer[BUFFER_SIZE];
	char send_buffer[BUFFER_SIZE];

	bzero(recv_buffer, BUFFER_SIZE);
	bzero(send_buffer, BUFFER_SIZE);

	while (1) {
		// Receive request to reset
		recvfrom(listenfd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr *) &serv_addr, &slen);

		//printf("Received packet at reset. Content: %s\n", recv_buffer);
		cooldown_amx_stage_4_tsc();
		
		// Send response back
		sendto(listenfd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&serv_addr, slen);
	}
}

void* leak_function() {
	int listenfd = 0;
	struct sockaddr_in serv_addr, cli_addr;
	socklen_t slen = sizeof(serv_addr);

	listenfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	memset(&serv_addr, 0, sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(LEAK_PORT);

	bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

	char recv_buffer[BUFFER_SIZE];
	char send_buffer[BUFFER_SIZE];

	bzero(recv_buffer, BUFFER_SIZE);
	bzero(send_buffer, BUFFER_SIZE);

	while (1) {
		// Receive packet to leak 
		recvfrom(listenfd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&cli_addr, &slen);

		//printf("Received packet at leak. Content: %s\n", recv_buffer);
		
		// Send response back
		sendto(listenfd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&cli_addr, slen);
	}	
}

// Reads from the variable that is either cached or uncached, responds
void* recover_function() {
	int listenfd = 0;
	struct sockaddr_in serv_addr, cli_addr;
	socklen_t slen = sizeof(serv_addr);

	listenfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	memset(&serv_addr, 0, sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(RECOVER_PORT);

	bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

	char recv_buffer[BUFFER_SIZE];
	char send_buffer[BUFFER_SIZE];

	bzero(recv_buffer, BUFFER_SIZE);
	bzero(send_buffer, BUFFER_SIZE);

	init_amx_all_tiles_max();
	
	while (1) {
		// Receive packet to recover
		recvfrom(listenfd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&cli_addr, &slen);

		//TIMER_INIT();
		//TIMER_START();
		asm volatile("TDPBSSD %tmm0, %tmm1, %tmm2\n");
		//TIMER_END();

		//printf("%"PRIu64"\n", TIMER_VALUE());

		//printf("Received packet at recover. Content: %s\n", recv_buffer);
		
		// Send response back
		sendto(listenfd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&cli_addr, slen);
	}
}

int main() {
	
	pthread_t reset_thread, recover_thread, leak_thread;
	
    pthread_create(&reset_thread, NULL, reset_function, NULL);
	pthread_create(&leak_thread, NULL, leak_function, NULL);
	pthread_create(&recover_thread, NULL, recover_function, NULL);
	
	pthread_join(reset_thread, NULL);
	pthread_join(recover_thread, NULL);
	pthread_join(leak_thread, NULL);
}
