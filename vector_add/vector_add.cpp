#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define STRING_BUFFER_LEN 1000
using namespace std;

const char *getErrorString(cl_int error);
const unsigned N = 50000000;

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

void checkError(int status, const char *msg) {
    if (status != CL_SUCCESS) {
        printf("%s: %s\n", msg, getErrorString(status));
        exit(0);
    }
}

void display_time(struct timespec *start, struct timespec *end,
                  const char *name) {
    double time = (double)(end->tv_sec - start->tv_sec) +
                  (double)(end->tv_nsec - start->tv_nsec) / 1000000000.;
    double flops = (((double)N) / time) / 1000000.;
    printf("%s took %.9lf seconds (%.lf Mflops).\n", name, time, flops);
}

void auto_display_time(struct timespec *start, const char *name) {
    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    display_time(start, &end, name);
    clock_gettime(CLOCK_REALTIME, start);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() { return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f; }

int main() {
    char char_buffer[STRING_BUFFER_LEN];
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, 0, 0};
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    //--------------------------------------------------------------------

    float *input_a = 0;
    float *input_b = 0;
    float *output = 0;
    struct timespec memory_start, memory_end;
    double memory_time = 0;
#ifndef MAPPED
    clock_gettime(CLOCK_REALTIME, &memory_start);
    input_a = (float *)malloc(N * sizeof(float));
    input_b = (float *)malloc(N * sizeof(float));
    output = (float *)malloc(N * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &memory_end);
    memory_time +=
        (double)(memory_end.tv_sec - memory_start.tv_sec) +
        (double)(memory_end.tv_nsec - memory_start.tv_nsec) / 1000000000.;
#endif
    float *ref_output = (float *)malloc(sizeof(float) * N);
    cl_mem input_a_buf;
    cl_mem input_b_buf;
    cl_mem output_buf;
    cl_event write_event[2];
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
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkError(status, "get device id failed");
    context =
        clCreateContext(context_properties, 1, &device, NULL, NULL, &status);
    checkError(status, "context creation failed");
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
    checkError(status, "command queue creation failed");

    unsigned char **opencl_program = read_file("vector_add.cl");
    program = clCreateProgramWithSource(
        context, 1, (const char **)opencl_program, NULL, &status);
    free(opencl_program);
    checkError(status, "program creation failed");

    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    checkError(status, "building program failed");
    kernel = clCreateKernel(program, "vector_add", &status);
    checkError(status, "kernel creation failed");

    // Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float),
                                 NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float),
                                 NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float),
                                NULL, &status);
    checkError(status, "Failed to create buffer for output");

    auto_display_time(&start, "GPU context init");

    clock_gettime(CLOCK_REALTIME, &memory_start);

#ifdef MAPPED
    input_a = (float *)clEnqueueMapBuffer(
        queue, input_a_buf, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0,
        N * sizeof(float), 0, NULL, &write_event[0], &status);
    checkError(status, "enqueue map buffer failed");
    input_b = (float *)clEnqueueMapBuffer(
        queue, input_b_buf, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0,
        N * sizeof(float), 0, NULL, &write_event[1], &status);
    checkError(status, "enqueue map buffer failed");

    clWaitForEvents(2, write_event);

    auto_display_time(&start, "GPU map buffer");
#endif

    // RANDOM NUMBER GENERATION
    for (unsigned j = 0; j < N; ++j) {
        input_a[j] = rand_float();
        input_b[j] = rand_float();
    }

    auto_display_time(&start, "random number generation");

    // CPU ADDITION
    for (unsigned j = 0; j < N; ++j) {
        ref_output[j] = input_a[j] + input_b[j];
    }

    auto_display_time(&start, "CPU addition");

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is
    // used for the host-to-device transfer.

    cl_event kernel_event;
#ifndef MAPPED
    status =
        clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE, 0, N * sizeof(float),
                             input_a, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status =
        clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE, 0, N * sizeof(float),
                             input_b, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");
    auto_display_time(&start, "GPU writes");
#else

    status = clEnqueueUnmapMemObject(queue, input_a_buf, input_a, 0, NULL,
                                     &write_event[0]);
    checkError(status, "enqueue unmap failed");
    status = clEnqueueUnmapMemObject(queue, input_b_buf, input_b, 0, NULL,
                                     &write_event[1]);
    checkError(status, "enqueue unmap failed");
    clWaitForEvents(2, write_event);
    auto_display_time(&start, "unmap buffers");
#endif

    clock_gettime(CLOCK_REALTIME, &memory_end);
    memory_time +=
        (double)(memory_end.tv_sec - memory_start.tv_sec) +
        (double)(memory_end.tv_nsec - memory_start.tv_nsec) / 1000000000.;

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

    clWaitForEvents(2, write_event);

    auto_display_time(&start, "GPU data copy");

    // GPU RUN

    const size_t global_work_size = N;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size,
                                    NULL, 0, NULL, &kernel_event);
    checkError(status, "Failed to launch kernel");
    // Read the result. This the final operation.
    clWaitForEvents(1, &kernel_event);

    auto_display_time(&start, "GPU run");

    // GPU READ

    clock_gettime(CLOCK_REALTIME, &memory_start);

#ifndef MAPPED
    status =
        clEnqueueReadBuffer(queue, output_buf, CL_FALSE, 0, N * sizeof(float),
                            output, 0, NULL, &kernel_event);
    checkError(status, "enqueue read buffer failed");
    clWaitForEvents(1, &kernel_event);

    auto_display_time(&start, "GPU read output");
#else
    output =
        (float *)clEnqueueMapBuffer(queue, output_buf, CL_FALSE, CL_MAP_READ, 0,
                                    N * sizeof(float), 0, 0, NULL, &status);
    checkError(status, "enqueue map buffer failed");

    auto_display_time(&start, "GPU output map");
#endif

    // VALUES CHECK

    // Verify results.
    bool pass = true;

    for (unsigned j = 0; j < N && pass; ++j) {
        if (fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
            printf(
                "Failed verification @ index %d\nOutput: %f\nReference: %f\n",
                j, output[j], ref_output[j]);
            pass = false;
        }
    }

    auto_display_time(&start, "value verification");

    clock_gettime(CLOCK_REALTIME, &memory_end);
    memory_time +=
        (double)(memory_end.tv_sec - memory_start.tv_sec) +
        (double)(memory_end.tv_nsec - memory_start.tv_nsec) / 1000000000.;

    printf("%s memory operations (read, write, ...) took %0.9lf seconds\n",
#ifdef MAPPED
           "mapped"
#else
           "copied"
#endif
           ,
           memory_time);

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(input_a_buf);
    clReleaseMemObject(input_b_buf);
    clReleaseMemObject(output_buf);
    clReleaseProgram(program);
    clReleaseContext(context);

    //--------------------------------------------------------------------

    clFinish(queue);

    return 0;
}

// from
// https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char *getErrorString(cl_int error) {
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
