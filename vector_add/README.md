# vector addition
## Description
The objective was to write a function in OpenCL to add two vectors, and to compare runtimes between CPU and GPU runs.

## Parameters
- `MAPPED`: if set, maps memory from kernel space to user space, else copies memory

## Statistics
Here are the time statistics depending on the size of the vector, and on whether we're using the cpu, or if we're using the gpu whether the memory was mapped or copied. Times are in microseconds.

|   N    | 1000  | 10000 | 100000 | 1000000 | 10000000 | 50000000 |
|--------|-------|-------|------- |-------  |-------   |----------|
|  CPU   | 0.6   | 30    | 228    | 1892    | 18017    | 80742    |
|  GPU   | 24    | 26    | 62     | 330     | 907      | 3357     |
| mapped | 1180  | 1537  | 4967   | 43068   | 410215   | 1961382  |
| copied | 348   | 744   | 4046   | 38027   | 371854   | 1814686  |

The lines `mapped` and `copied` give the time taken by all memory related operations: allocation, reading, writing, (un)mapping and copying.

Clearly the bigger N the more the GPU beats the CPU. We can notice that for small N the CPU is much faster than the GPU,

Weirly mapped memory is not faster than copied one. I'm quite sure it was faster on the odroid with the same program.
