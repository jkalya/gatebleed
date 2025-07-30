#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h> //inet_addr
#include <sched.h>
#include <time.h>
#include <stdlib.h>
#include "../timer.h" // change when copying
#include "network_config.h" // change when copying
#include "../amx.h"
#include <poll.h>

// #define SERVER_ADDRESS "127.0.0.1" // CHANGE THIS
#define NUM_TRIALS 10000

int main(int argc, char* argv[]) {
	if (argc != 2) {
		perror("Syntax: ./client IP_ADDRESS");
	}

	char SERVER_ADDRESS[100];
	strncpy(SERVER_ADDRESS, argv[1], 16);
	

	int reset_fd, recover_fd, leak_fd;
	struct sockaddr_in reset_socket, recover_socket, leak_socket;
	int reset_socklen, recover_socklen, leak_socklen;
	
	char send_buffer[BUFFER_SIZE];
	char recv_buffer[BUFFER_SIZE];

	reset_fd=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
	recover_fd=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
	leak_fd=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);

	bzero(&reset_socket, sizeof(reset_socket));
    bzero(&recover_socket, sizeof(recover_socket));
	bzero(&leak_socket, sizeof(leak_socket));

	reset_socket.sin_addr.s_addr = inet_addr(SERVER_ADDRESS);
    reset_socket.sin_family=AF_INET;
    reset_socket.sin_port=htons(RESET_PORT);

    recover_socket.sin_addr.s_addr = inet_addr(SERVER_ADDRESS);
    recover_socket.sin_family=AF_INET;
    recover_socket.sin_port=htons(RECOVER_PORT);

	leak_socket.sin_addr.s_addr = inet_addr(SERVER_ADDRESS);
    leak_socket.sin_family=AF_INET;
    leak_socket.sin_port=htons(LEAK_PORT);

	reset_socklen = sizeof(reset_socket);
	recover_socklen = sizeof(recover_socket);
	leak_socklen = sizeof(leak_socket);

	//TIMER_INIT();

	for (int i = 0; i < NUM_TRIALS; i++) {

		// Send to reset endpoint
		// strcpy(send_buffer, "RESET");
		// sendto(reset_fd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, sizeof(reset_socket));
		// recvfrom(reset_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, &reset_socklen);

		// "ping" the server (see comment in server.c)
		strcpy(send_buffer, "LEAK");
		sendto(leak_fd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&leak_socket, sizeof(leak_socket));
		TIMER_START();
		recvfrom(leak_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&leak_socket, &leak_socklen);
		TIMER_END();
		int64_t ping_time = TIMER_VALUE();


		TIMER_INIT();

		// Send to recover endpoint
		strcpy(send_buffer, "RECOVER"); 
		sendto(recover_fd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&recover_socket, sizeof(recover_socket));
		TIMER_START();
		recvfrom(recover_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&recover_socket, &recover_socklen);
		TIMER_END();

		printf("0,%"PRId64"\n", (int64_t)TIMER_VALUE()-ping_time);
	}

	for (int i = 0; i < NUM_TRIALS; i++) {

		// Send to reset endpoint
		/* strcpy(send_buffer, "RESET"); */
		/* sendto(reset_fd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, sizeof(reset_socket)); */
		/* recvfrom(reset_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, &reset_socklen); */

		usleep(30000);

		//WARMUP();

		// "ping" the server (see comment in server.c)
		strcpy(send_buffer, "LEAK");
		sendto(leak_fd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&leak_socket, sizeof(leak_socket));
		TIMER_START();
		recvfrom(leak_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&leak_socket, &leak_socklen);
		TIMER_END();
		int64_t ping_time = TIMER_VALUE();


		TIMER_INIT();

		// Send to recover endpoint
		strcpy(send_buffer, "RECOVER"); 
		sendto(recover_fd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&recover_socket, sizeof(recover_socket));
		TIMER_START();
		recvfrom(recover_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&recover_socket, &recover_socklen);
		TIMER_END();

		printf("1,%"PRId64"\n", (int64_t)TIMER_VALUE()-ping_time);
	}
	//printf("%"PRIu64"\n", TIMER_VALUE());
	
	return 0;
}
