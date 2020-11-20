# intropencl
An introduction to OpenCL (Lutecia Week 2020 TP39).

Course from [https://github.com/amusant/tpt39](https://github.com/amusant/tpt39).

## Device
The target was supposed to be an `odroid xu4` (with an `ARM Mali T628` GPU), but I changed a few things to have it run on my own computer (whose GPU is an `Nvidia GTX 1070`).

It should run on any nvidia card and any Linux with Nvidia and OpenCL libs installed.

If you want to run it on another device you probably only need to change the libs in the makefile. You can also change the version of `C++` used.

## Usage
Each subdirectory has a `Makefile`, running `make` compiles and run the `.cpp` file (which itself compiles and run the `.cl` file).

## Hello world
Writes "hello world !", it's mainly just to see how to setup an OpenCL program.

## Vector add
Adds two vectors together, compares times between CPU and GPU.

If `MAPPED` is defined, the memory used is mapped from GPU space to CPU space.

If not, the memory is copied.

## Matrix product
Multiplies two matrix together, and compares times between CPU and GPU.

If `GROUPS` is defined, it uses groups to have cache speedup.

## Video filter
Video filtering calculations in GPU (opencv stuff) to modify a video.

In particular write a function to highlight contours in an image and one to blur an image.

You can define `GROUPS` and `MAPPED` to use groups and to map memory instead of copy it.

Compare runtimes between OpenCV and home made OpenCL functions.
