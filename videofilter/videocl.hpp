#ifndef VIDEOCL_H
#define VIDEOCL_H

#include <CL/cl.h>
#include <CL/cl_ext.h>

void init();
void ScharrCL();
void addWeightedCL();
void thresholdCL();

#endif
