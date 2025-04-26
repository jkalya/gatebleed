# remote_gatebleed_spectre

This is a remote Spectre-v1 attack using the gatebleed covert channel for increased visibility over a noisy channel such as a network

Steps
- Open client.c and update SERVER_ADDRESS to the ip address of the server (it is 127.0.0.1 by default)
- `make`
- On the server, run `./server`. It will print out the starting index that we can use to attack. 
For example, `./server` can output this:
```
Starting point: -229415
Secret: It was the best of times, it was the worst of times
```

- On the client, run `./client INDEX` which we obtained from running `server`. In the above example, you start with -229415 and increase. Every 8 bits leaked is a byte

 
