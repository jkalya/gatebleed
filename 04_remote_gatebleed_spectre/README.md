# remote_gatebleed_spectre

This is a remote Spectre-v1 attack using the gatebleed covert channel for increased visibility over a noisy channel such as a network
Note - make sure `03_remote_gatebleed_covert_channel` is working before trying this. 

## Building
### Client
- Open client.c and update SERVER_ADDRESS to the ip address of the server (it is 127.0.0.1 by default)
- Modify networking_config.h if necessary to change ports
- `make`

### Server
- Modify networking_config.h if necessary to change ports

## Running
### Server
- On the server, run `./server`. It will print out the starting index that we can use to attack. 
For example, `./server` can output this:
```
Starting point: -229415
Secret: It was the best of times, it was the worst of times
```
- On the client, run `./leak_byte.bash STARTING_POINT` which we obtained from running `server`. In the above example, start with -229415 and increase by 8 for each byte.
- This produces a csv file called data\_STARTING\_POINT.csv

### Plotting
We then plot the byte, along with if the bit was classified 0 or 1. 
Do `./plot_byte.py <CSV> <MIDPOINT>` 
where CSV is the generated file in the running step 
MIDPOINT is obtained from artifact 03 as a reasonable dividing line between AMX cold and hot based on the data collected in that experiment
For example, we found on our 1-hop network that a reasonable dividing line is 19924. This generates `<CSV>.png` which can be compared to the ASCII representation of the leaked text. 
The leaked text in server.c is "It was the best of times, it was the worst of times" so byte 0 should yield I="01001001"

### Troubleshooting
First, double check the dividing line is good and update if necessary.
Attack efficacy is highly dependent on the network noise. Increase the number of trials performed by the client in client.c by increasing the NUMBER\_OF\_TRIALS macro on line 126 (It is currently set to 500) 
This attack works best when C-states are disabled and C1e is disabled. 
If disabling C-states and C1e isn't possible but SMT is enabled, run ./busyloop on the shared sibling hyperthread of the ./server process to keep the CPU awake
