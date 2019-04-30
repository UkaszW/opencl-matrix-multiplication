//
//  main.cpp
//  opencl-matrix-multiplication
//
//  Created by Łukasz Wójcik on 30/04/2019.
//  Copyright © 2019 Łukasz Wójcik. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CL_SILENCE_DEPRECATION

#define WA 1024
#define HA 1024
#define WB 1024
#define HB WA
#define WC WB
#define HC HA

using namespace std;

//---------------------------------------------------

const char *getErrorString(cl_int error)
{
    switch (error){
            // run-time and JIT compiler errors
        case  0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

char*  source_buf;
size_t source_size;

void checkCL(cl_int status)
{
    if (status != CL_SUCCESS)
    {
        cout << "OpenCL Error       : " << getErrorString(status) << endl;
    }
}

//---------------------------------------------------

const unsigned int N = 1024;

float h_buf1[N];
float h_buf2[N];
float h_buf3[N];

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    srand(0);
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void identityMultipliedByCoeff(float* data, int width, float coeff)
{
    for (int i = 0; i <width*width; ++i)
    {
        data[i] = 0.0;
    }
    for (int i = 0; i < width; ++i)
    {
        data[i*width + i] = 1.0*coeff;
    }
}

int main()
{
    char str[1024];
    size_t                size;
    cl_int                ret;
    cl_uint                tmp;
    cl_ulong            utmp;
    cl_ulong            time_end;
    cl_ulong            time_start;
    cl_uint                num_devices;
    cl_uint                num_platforms;
    cl_mem                mem1;
    cl_mem                mem2;
    cl_mem                mem3;
    cl_event            time_event;
    cl_kernel            kernel;
    cl_context            context;
    cl_program            program;
    cl_device_id        device_id;
    cl_platform_id        platform_id;
    cl_command_queue    command_queue;
    size_t                num_of_globals;
    size_t                num_of_locals;
    FILE*                fp;
    
    // 1. allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
    
    // 2. initialize host memory
    //randomInit(h_A, size_A);
    //randomInit(h_B, size_B);
    identityMultipliedByCoeff(h_A, WA, 12.0);
    identityMultipliedByCoeff(h_B, WB, -9.0);
    
    // 4. allocate host memory for the result C
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);
    
    fp = fopen( "/Users/lukaszwojcik/Development/opencl-matrix-multiplication/opencl-matrix-multiplication/matrix_mul_kernel.cl", "r");
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        source_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        source_buf = (char*)malloc(source_size);
        fread(source_buf, 1, source_size, fp);
        fclose(fp);
    } else {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    // Data prepare
    for (int i = 0; i<N; i++)
    {
        h_buf1[i] = (float)i;
        h_buf2[i] = (float)(N - i);
    }
    // Init
    checkCL(clGetPlatformIDs(1, &platform_id, &num_platforms));
    // Platform info.
    checkCL(clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 1024, str, &size));
    cout << "Platform name      : " << str << endl;
    // Device info.
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 2, &device_id, &num_devices);
    cl_device_type t;
    checkCL(clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(t), &t, &size));
    if (t == CL_DEVICE_TYPE_CPU)
    {
        cout << "Device type        : CPU" << endl;
    }
    if (t == CL_DEVICE_TYPE_GPU)
    {
        cout << "Device type        : GPU" << endl;
    }
    checkCL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(tmp), &tmp, &size));
    cout << "Number of units    : " << tmp << endl;
    checkCL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(utmp), &utmp, &size));
    cout << "Max. memory alloc. : " << utmp / (1024 * 1024) << " MB" << endl;
    checkCL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(utmp), &utmp, &size));
    cout << "Global mem. size   : " << utmp / (1024 * 1024) << " MB" << endl;
    checkCL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(tmp), &tmp, &size));
    cout << "Clock frequency    : " << tmp << " MHz " << endl;
    // ---
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    checkCL(ret);
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    checkCL(ret);
    mem1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &ret);
    checkCL(ret);
    mem2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &ret);
    checkCL(ret);
    mem3 = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_C, NULL, &ret);
    checkCL(ret);
    program = clCreateProgramWithSource(context, 1, (const char**)&source_buf, (const size_t*)&source_size, &ret);
    checkCL(ret);
    checkCL(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
    kernel = clCreateKernel(program, "matrixMul", &ret);
    checkCL(ret);
    // Arguments
    int wA = WA;
    int wB = WB;
    int wC = WC;
    checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem1));
    checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem2));
    checkCL(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem3));
    checkCL(clSetKernelArg(kernel, 3, sizeof(int), (void *)&wA));
    checkCL(clSetKernelArg(kernel, 4, sizeof(int), (void *)&wC));
    // Start kernel
    num_of_locals = 16;
    num_of_globals = 1024;
    checkCL(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &num_of_globals, &num_of_locals, 0, NULL, &time_event));
    /*checkCL(clWaitForEvents(1, &time_event));
    checkCL(clGetEventProfilingInfo(time_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL));
    checkCL(clGetEventProfilingInfo(time_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL));
    cout << "TT                 : " << (time_end - time_start) / 1000 << " us" << endl;
    checkCL(clEnqueueReadBuffer(command_queue, mem3, CL_TRUE, 0,  mem_size_C, h_C, 0, NULL, NULL));
    for (int j = 0; j < N; j++) {
        printf("%f + %f = %f\n", h_buf1[j], h_buf2[j], h_buf3[j]);
    }
    */
    
    // 8. Retrieve result from device
    checkCL(clEnqueueReadBuffer(command_queue, mem3, CL_TRUE, 0,  mem_size_C, h_C, 0, NULL, NULL));
    
    // We must check the result
    for (int i = 0; i < WA; i++)
    {
        for (int j = 0; j < WA; j++)
        {
            float prod = 0;
            for (int k = 0; k < WA;k++)
            {
                prod += h_A[i*WA + k] * h_B[k*WA + j];
            }
            if (fabs(h_C[i*WA+j] - prod) > 0.01)
            {
                printf("The indices where the comparison failed, i = %d, j = %d\n", i,j);
                printf("C[i*WA+j] should equal %f\n", prod);
                printf("C[i*WA+j] = %f\n", h_C[i*WA + j]);
                perror("The matrix check has failed");
                exit(1);
                break;
            }
        }
    }
    printf("The matrix check has been successfull!\n");
    
    // Finish
    free(h_A);
    free(h_B);
    free(h_C);
    checkCL(clFlush(command_queue));
    checkCL(clFinish(command_queue));
    checkCL(clReleaseKernel(kernel));
    checkCL(clReleaseProgram(program));
    checkCL(clReleaseMemObject(mem1));
    checkCL(clReleaseMemObject(mem2));
    checkCL(clReleaseMemObject(mem3));
    checkCL(clReleaseCommandQueue(command_queue));
    checkCL(clReleaseContext(context));
    free(source_buf);
    return 0;
}
