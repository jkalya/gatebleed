WARMUP_AMX=1

all: server client

.PHONY: server client

server:
	gcc -O0 -march=native -DWARMUP_AMX=$(WARMUP_AMX) server.c -o server

server_asm:
	gcc -S -O0 -march=native -DWARMUP_AMX=$(WARMUP_AMX) server.c -o server.asm

client:
	gcc -O0 -march=native client.c -o client
