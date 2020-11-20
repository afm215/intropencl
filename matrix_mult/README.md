# matrix multiplication
## Description
The objective was to write a function in OpenCL to multiply two matrices, and to compare runtimes between CPU and GPU runs.

## Parameters
- `GROUPS=g`: if set, groups are used to launch threads, the groups are of size gxg

## Statistics
Here are the time statistics depending on the size of the matrix (we only consider square matrices here), on whether we're using the cpu, or if we're using the gpu, what size of group we're using (none = no group). Times are in seconds.

|   N    | 100     | 500     | 1000    | 2000    | 5000   |
|--------|---------|---------|---------|---------|--------|
|  cpu   | 0.0004  | 0.0539  | 0.51889 | 10.1067 |  ++++  |
|  none  | 0.00007 | 0.00281 | 0.0207  | 0.17807 | 2.6598 |
|   2    | 0.00023 | 0.02263 | 0.17688 | 1.12390 | 18.395 |
|   4    | 0.00009 | 0.00581 | 0.04561 | 0.3385  | 4.3886 |
|   5    | 0.00007 | 0.00385 | 0.02972 | 0.2305  | 3.1894 |
|   10   | 0.00006 | 0.00377 | 0.02865 | 0.1852  | 2.9327 |

We can notice that for small matrices the CPU is faster than the GPU, but the bigger the matrices the more the CPU seems slow compared to the GPU.

We can also see that small groups take much more time that big ones.

We can also see that in this example, not using groups is faster than using one, whatever the size of the group. I assume it as so much things to load that the cache optimizations are not important enough to balance the cost of groups.

Maybe for a bigger N and a big group size it could work ?
