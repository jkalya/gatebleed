# remote_gatebleed_covert_channel
Shows how the gatebleed covert channel is easily visible over the network. 

## Building
### Client
- Modify networking_config.h to change the ports if necessary
- `make client`

### Server
- Modify networking_config.h to change the ports if necessary
- `make server`

## Running
- On the server side, pin the task to a particular core, e.g. with `taskset -c 0 ./server`
- On the client side, run `./client IP_ADDRESS > data.csv`, e.g. IP_ADDRESS can be 127.0.0.1 if testing localhost
- To generate the plot, run `python3 plot.py`
- The plot is saved as plot.png

## Troubleshooting
The plot should look like plot_original.png on localhost, i.e. client and server are same machine
If not, uncomment the lines `TIMER_INIT()`, `TIMER_START()`, `TIMER_END()`, and `printf("%"PRIu64"\n", TIMER_VALUE());` and rerun
This will have the server dump the execution time of the AMX instruction induced by the client to stdout. Make sure that the timings are significantly different, e.g. around 6000 for AMX warm and 20,000 for AMX cold.  
If the timings are not very different, disable C1 enhanced and C states in the BIOS.
An alternative to this is running `./busyloop` on the sibling hyperthread to `./server`'s thread. This prevents the core from entering any sleep states. 
For example, if server was run with `taskset -c 28 ./server` and the server is a 56-core machine, run `taskset -c 84 ./busyloop`. This requires Hyperthreading be enabled. 
