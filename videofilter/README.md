# videofilter
## Description
The objective was to rewrite two OpenCV functions with OpenCL and compare runtimes.
The two functions are `Scharr`, which brings to light contours and `GaussianBlur`, which blurs the image.

## Parameters
- `OPENCV`: if set, uses functions from the OpenCV library, else uses OpenCL functions

When OPENCV is not set:
- `MAPPED`: if set, maps memory from kernel space to user space, else copies memory
- `GROUPS=g`: if set, groups are used to launch threads, the groups are of size gxg

## Statistics
Here are the time statistics depending on the size of the group, and on whether the memory was mapped or copied. The column with size "none" is without using groups.

|  size  | none  |   2   |   4   |   8   |
|--------|-------|-------|-------|-------|
| mapped | 0.14s | 0.37s | 0.19s | 0.15s |
| copy   | 0.16s | 0.40s | 0.21s | 0.17s |

I assume the groups are not a such a good idea here, each thread probably uses too much memory for the cache to be really useful.

As a comparison using OpenCV takes 0.60s.

## Enhancements
I also rewrote `addWeighted`.

I avoid copying the image, and I don't copy it to add the border, to reduce runtimes as much as possible.
