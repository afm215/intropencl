#include "videocl.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define STRING_BUFFER_LEN 1024

static const char *getErrorString(cl_int error);
static void checkError(int status, const char *msg);
static const char *getErrorString(cl_int error);
static unsigned char **read_file(const char *name);

static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel_convol, kernel_add;

static const cl_int scharrh[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
static const cl_int scharrv[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

void ScharrCL(void *frame, void *output, size_t width, size_t height,
              bool horizontal) {
    // Set kernel arguments.
    static const size_t local_size = 4;
    unsigned argi = 0;
    cl_int status, W = width, H = height, D = 3;
    cl_event event;

    status = clSetKernelArg(kernel_convol, argi++, sizeof(cl_mem), frame);
    checkError(status, "[SCHARR] kernel set arg 1 failed");

    status = clSetKernelArg(kernel_convol, argi++, sizeof(cl_mem),
                            horizontal ? scharrh : scharrv);
    checkError(status, "[SCHARR] kernel set arg 2 failed");

    status = clSetKernelArg(kernel_convol, argi++, sizeof(cl_mem), output);
    checkError(status, "[SCHARR] kernel set arg 3 failed");

    status = clSetKernelArg(kernel_convol, argi++, sizeof(cl_uint), &W);
    checkError(status, "[SCHARR] kernel set arg 4 failed");

    status = clSetKernelArg(kernel_convol, argi++, sizeof(cl_uint), &H);
    checkError(status, "[SCHARR] kernel set arg 5 failed");

    status = clSetKernelArg(kernel_convol, argi++, sizeof(cl_uint), &D);
    checkError(status, "[SCHARR] kernel set arg 6 failed");

    // GPU RUN
    const size_t global_work_size[] = {width, height};
    const size_t local_work_size[] = {local_size, local_size};
    status =
        clEnqueueNDRangeKernel(queue, kernel_convol, 2, NULL, global_work_size,
                               local_work_size, 0, NULL, &event);
    checkError(status, "[SCHARR] Failed to launch kernel");
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
}

void averageCL(void *input1, void *input2, void *output, size_t width,
               size_t height) {
    // Set kernel arguments.
    static const size_t local_size = 4;
    unsigned argi = 0;
    cl_int status;
    cl_event event;

    status = clSetKernelArg(kernel_add, argi++, sizeof(cl_mem), input1);
    checkError(status, "[AVERAGE] kernel set arg 1 failed");

    status = clSetKernelArg(kernel_add, argi++, sizeof(cl_mem), input2);
    checkError(status, "[AVERAGE] kernel set arg 2 failed");

    status = clSetKernelArg(kernel_add, argi++, sizeof(cl_mem), output);
    checkError(status, "[AVERAGE] kernel set arg 3 failed");

    // GPU RUN
    const size_t global_work_size = width * height;
    status = clEnqueueNDRangeKernel(queue, kernel_add, 1, NULL,
                                    &global_work_size, NULL, 0, NULL, &event);
    checkError(status, "[AVERAGE] Failed to launch kernel");
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
}

void cl_memwrite(void *from, void *to, size_t size) {
    cl_int status = clEnqueueWriteBuffer(queue, (cl_mem)to, CL_TRUE, 0, size,
                                         from, 0, NULL, NULL);
    checkError(status, "Failed to transfer input A");
}

void cl_memread(void *from, void *to, size_t size) {
    cl_int status = clEnqueueReadBuffer(queue, (cl_mem)to, CL_TRUE, 0, size,
                                        from, 0, NULL, NULL);
    checkError(status, "Failed to transfer input A");
}

void thresholdCL(void *input, void *output, size_t width, size_t height) {
    // TODO
}

void *cl_getmem(size_t size) {
    cl_int status;
    auto buff = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &status);
    checkError(status, "Failed to create buffer for input A");
    return buff;
}

void *cl_map_mem(void *buffer, size_t size) {
    cl_event event;
    cl_int status;
    void *ret = clEnqueueMapBuffer(queue, (cl_mem)buffer, CL_TRUE,
                                   CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL,
                                   &event, NULL);
    checkError(status, "enqueue map failed");
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    return ret;
}

void cl_unmap_mem(void *from, void *to) {
    cl_event event;
    cl_int status =
        clEnqueueUnmapMemObject(queue, (cl_mem)from, to, 0, NULL, &event);
    checkError(status, "enqueue unmap failed");
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
}

void cl_init() {
    char char_buffer[STRING_BUFFER_LEN];
    cl_platform_id platform;
    cl_device_id device;
    cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, 0, 0};

    //--------------------------------------------------------------------
    int status;

    // GPU CONTEXT INIT
    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN,
                      char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN,
                      char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN,
                      char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

    context_properties[1] = (cl_context_properties)platform;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context =
        clCreateContext(context_properties, 1, &device, NULL, NULL, &status);
    checkError(status, "context creation failed");
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
    checkError(status, "command queue creation failed");

    unsigned char **opencl_program = read_file("matrix_prod.cl");
    program = clCreateProgramWithSource(
        context, 1, (const char **)opencl_program, NULL, &status);
    checkError(status, "program creation failed");
    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    checkError(status, "program build failed");

    kernel_convol = clCreateKernel(program, "convolution", &status);
    checkError(status, "kernel creation failed");
    kernel_add = clCreateKernel(program, "matrix_prod_groups", &status);
    checkError(status, "kernel creation failed");
}

void cl_releasemem(cl_mem buff) { clReleaseMemObject(buff); }

void cl_clean() {
    clReleaseKernel(kernel_convol);
    kernel_convol = 0;
    clReleaseKernel(kernel_add);
    kernel_add = 0;
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    clFinish(queue);
}

//
// helper functions
//

static void checkError(int status, const char *msg) {
    if (status != CL_SUCCESS) {
        printf("%s: %s\n", msg, getErrorString(status));
        exit(0);
    }
}

// from
// https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
static const char *getErrorString(cl_int error) {
    switch (error) {
    // run-time and JIT compiler errors
    case 0:
        return "CL_SUCCESS";
    case -1:
        return "CL_DEVICE_NOT_FOUND";
    case -2:
        return "CL_DEVICE_NOT_AVAILABLE";
    case -3:
        return "CL_COMPILER_NOT_AVAILABLE";
    case -4:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:
        return "CL_OUT_OF_RESOURCES";
    case -6:
        return "CL_OUT_OF_HOST_MEMORY";
    case -7:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8:
        return "CL_MEM_COPY_OVERLAP";
    case -9:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case -10:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11:
        return "CL_BUILD_PROGRAM_FAILURE";
    case -12:
        return "CL_MAP_FAILURE";
    case -13:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case -16:
        return "CL_LINKER_NOT_AVAILABLE";
    case -17:
        return "CL_LINK_PROGRAM_FAILURE";
    case -18:
        return "CL_DEVICE_PARTITION_FAILED";
    case -19:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30:
        return "CL_INVALID_VALUE";
    case -31:
        return "CL_INVALID_DEVICE_TYPE";
    case -32:
        return "CL_INVALID_PLATFORM";
    case -33:
        return "CL_INVALID_DEVICE";
    case -34:
        return "CL_INVALID_CONTEXT";
    case -35:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case -36:
        return "CL_INVALID_COMMAND_QUEUE";
    case -37:
        return "CL_INVALID_HOST_PTR";
    case -38:
        return "CL_INVALID_MEM_OBJECT";
    case -39:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40:
        return "CL_INVALID_IMAGE_SIZE";
    case -41:
        return "CL_INVALID_SAMPLER";
    case -42:
        return "CL_INVALID_BINARY";
    case -43:
        return "CL_INVALID_BUILD_OPTIONS";
    case -44:
        return "CL_INVALID_PROGRAM";
    case -45:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46:
        return "CL_INVALID_KERNEL_NAME";
    case -47:
        return "CL_INVALID_KERNEL_DEFINITION";
    case -48:
        return "CL_INVALID_KERNEL";
    case -49:
        return "CL_INVALID_ARG_INDEX";
    case -50:
        return "CL_INVALID_ARG_VALUE";
    case -51:
        return "CL_INVALID_ARG_SIZE";
    case -52:
        return "CL_INVALID_KERNEL_ARGS";
    case -53:
        return "CL_INVALID_WORK_DIMENSION";
    case -54:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case -55:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case -56:
        return "CL_INVALID_GLOBAL_OFFSET";
    case -57:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case -58:
        return "CL_INVALID_EVENT";
    case -59:
        return "CL_INVALID_OPERATION";
    case -60:
        return "CL_INVALID_GL_OBJECT";
    case -61:
        return "CL_INVALID_BUFFER_SIZE";
    case -62:
        return "CL_INVALID_MIP_LEVEL";
    case -63:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64:
        return "CL_INVALID_PROPERTY";
    case -65:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66:
        return "CL_INVALID_COMPILER_OPTIONS";
    case -67:
        return "CL_INVALID_LINKER_OPTIONS";
    case -68:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000:
        return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001:
        return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002:
        return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003:
        return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004:
        return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005:
        return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default:
        return "Unknown OpenCL error";
    }
}

unsigned char **read_file(const char *name) {
    size_t size;
    unsigned char **output = (unsigned char **)malloc(sizeof(unsigned char *));
    FILE *fp = fopen(name, "rb");
    if (!fp) {
        printf("no such file:%s", name);
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *output = (unsigned char *)malloc(size);
    unsigned char **outputstr =
        (unsigned char **)malloc(sizeof(unsigned char *));
    *outputstr = (unsigned char *)malloc(size);
    if (!*output) {
        fclose(fp);
        printf("mem allocate failure:%s", name);
        exit(-1);
    }

    if (!fread(*output, size, 1, fp))
        printf("failed to read file\n");
    fclose(fp);
    printf("file size %lu\n", size);
    printf("-------------------------------------------\n");
    snprintf((char *)*outputstr, size, "%s\n", *output);
    printf("%s\n", *outputstr);
    printf("-------------------------------------------\n");
    return outputstr;
}
