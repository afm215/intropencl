#ifndef VIDEOCL_H
#define VIDEOCL_H

#include <CL/cl.h>
#include <CL/cl_ext.h>

void cl_init();
void cl_clean();

void *cl_map_mem(void *buffer, size_t size);
void cl_unmap_mem(void *from, void *to);

void cl_memwrite(void *from, void *to, size_t size);
void cl_memread(void *from, void *to, size_t size);

void *cl_getmem(size_t size);
void cl_releasemem(cl_mem buff);

void ScharrCL(void *frame, void *output, size_t width, size_t height,
              bool horizontal);
void averageCL(void *input1, void *input2, void *output, size_t width,
               size_t height);
void thresholdCL(void *input, void *output, size_t width, size_t height);

#endif
