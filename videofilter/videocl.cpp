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

#ifdef GROUPS
#define PROGRAM "convol_prod_groups"
#else
#define PROGRAM "convol_prod"
#endif

static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;

static const cl_int scharrh[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
static const cl_int scharrv[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

void ScharrCL() {
    // TODO
}

void addWeightedCL() {
    // TODO
}

void thresholdCL() {
    // TODO
}

void init() {
    char char_buffer[STRING_BUFFER_LEN];
    cl_platform_id platform;
    cl_device_id device;
    cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, 0, 0};

    //--------------------------------------------------------------------
    int M = 1;
    float *input_a = 0;
    float *input_b = 0;

    cl_mem input_a_buf; // num_devices elements
    cl_mem input_b_buf; // num_devices elements
    cl_mem output_buf;  // num_devices elements
    cl_event events[2];
    int status;

    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);

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

    kernel = clCreateKernel(program, PROGRAM, NULL);
    // Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, M * sizeof(float),
                                 NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, M * sizeof(float),
                                 NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, M * sizeof(float),
                                NULL, &status);
    checkError(status, "Failed to create buffer for output");

    input_a = (float *)clEnqueueMapBuffer(
        queue, input_a_buf, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0,
        M * sizeof(float), 0, NULL, &events[0], &status);
    checkError(status, "enqueue map failed");
    input_b = (float *)clEnqueueMapBuffer(
        queue, input_b_buf, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0,
        M * sizeof(float), 0, NULL, &events[1], &status);
    checkError(status, "enqueue map failed");

    clWaitForEvents(2, events);

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is
    // used for the host-to-device transfer.

    clEnqueueUnmapMemObject(queue, input_a_buf, input_a, 0, NULL, &events[0]);
    clEnqueueUnmapMemObject(queue, input_b_buf, input_b, 0, NULL, &events[1]);
    clWaitForEvents(2, events);
    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

    const cl_uint Mint = M;
    status = clSetKernelArg(kernel, argi++, sizeof(cl_uint), &Mint);
    checkError(status, "Failed to set argument 3");

// GPU RUN
#ifdef GROUPS
    const size_t global_work_size[] = {M, M};
    const size_t local_work_size[] = {local_size, local_size};
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                    local_work_size, 0, NULL, &events[0]);
#else
    const size_t global_work_size = M;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size,
                                    NULL, 0, NULL, &events[1]);
#endif
    checkError(status, "Failed to launch kernel");
    clWaitForEvents(2, events);

    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);
}

void cl_releasemem(cl_mem buff) { clReleaseMemObject(buff); }

void cl_clean() {
    clReleaseKernel(kernel);
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
