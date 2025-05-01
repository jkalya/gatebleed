all: server client busyloop

.PHONY: server client busyloop

server:
	gcc -O0 -march=native -DWARMUP_AMX=$(WARMUP_AMX) server.c -o server

server_asm:
	gcc -S -O0 -march=native -DWARMUP_AMX=$(WARMUP_AMX) server.c -o server.asm

client:
	gcc -O0 -march=native client.c -o client
busyloop:
	gcc -O0 busyloop.c -o busyloop
