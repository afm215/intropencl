# intropencl
An introduction to OpenCL (Lutecia Week 2020 TP39)

## Hello world
Writes "hello world !", it's mainly just to see how to setup an OpenCL program.

## Vector add
Adds two vectors together, compares times between CPU and GPU.

If `MAPPED` is defined, the memory used is mapped from GPU space to CPU space.

If not, the memory is copied.

## Matrix product
Multiplies two matrix together, and compares times between CPU and GPU.

If `GROUPS` is defined, it uses groups to have cache speedup. You can change the size of the groups in `matrix_prod.cpp`.
