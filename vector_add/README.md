# vector addition
## Description
The objective was to write a function in OpenCL to add two vectors, and to compare runtimes between CPU and GPU runs.

## Parameters
- `MAPPED`: if set, maps memory from kernel space to user space, else copies memory

## Statistics
Here are the time statistics depending on the size of the vector, and on whether we're using the cpu, or if we're using the gpu whether the memory was mapped or copied. Times are in microseconds.

|   N    | 1000  | 10000 | 100000 | 1000000 | 10000000 | 50000000 |
|--------|-------|-------|------- |-------  |-------   |----------|
|  cpu   | 0.6   | 30    | 228    | 1892    | 18017    | 80742    |
| mapped | 24    | 26    | 62     | 330     | 907      | 3357     |
| copy   | 25    | 25    | 71     | 252     | 870      | 3379     |

Clearly the bigger N the more the GPU beats the CPU. We can notice that for small N the CPU is much faster than the GPU,

Weirly mapped memory is not faster than copied one. I'm quite sure it was faster on the odroid with the exact same program.
