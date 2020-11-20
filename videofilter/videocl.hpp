#ifndef VIDEOCL_H
#define VIDEOCL_H

#include <CL/cl.h>
#include <CL/cl_ext.h>

void cl_init();
void cl_clean();

void *cl_map_mem(cl_mem buffer, size_t size);
void cl_unmap_mem(cl_mem from, void *to);

void cl_memwrite(void *cpubuff, cl_mem gpubuff, size_t size);
void cl_memread(cl_mem gpubuff, void *cpubuff, size_t size);

cl_mem cl_getmem(size_t size);
void cl_releasemem(cl_mem buff);

void cl_blur(cl_mem frame, cl_mem output, size_t width, size_t height);
void cl_Scharr(cl_mem frame, cl_mem output, size_t width, size_t height,
               bool horizontal);
void cl_average(cl_mem input1, cl_mem input2, cl_mem output, size_t width,
                size_t height);

#endif
