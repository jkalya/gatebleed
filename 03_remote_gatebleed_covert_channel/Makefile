all: server client busyloop
.PHONY: server client busyloop
server:
	gcc -O0 -march=native server.c -o server
client:
	gcc -O0 -march=native client.c -o client
busyloop:
	gcc -O0 -march=native busyloop.c -o busyloop

clean:
	rm server client
