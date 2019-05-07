//
//  main.cpp
//  opencl-matrix-multiplication
//
//  Created by Łukasz Wójcik on 30/04/2019.
//  Copyright © 2019 Łukasz Wójcik. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK_STATUS( status, message )   \
if(status != CL_SUCCESS) \
{ \
printf( message); \
printf( "\n ERR %d\n", status  ); \
fflush(NULL);\
return 1; \
}

#define MATRIX_SIZE 512

#define MAX_SOURCE_SIZE (1000000)

#define CL_SILENCE_DEPRECATION

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    srand(0);
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

int main(int argc, char** argv)
{
    //------------------------------------------------------------
    //defining the input argument
    //------------------------------------------------------------
    cl_int numData = MATRIX_SIZE;
    
    printf("Matrix dimension : %d x %d : %d \n", numData, numData, numData*numData);
    
    //-------------------------------------------------------------
    //first we have to get the platform we have in hand
    //-------------------------------------------------------------
    cl_int ret;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_uint num_platforms;
    cl_uint num_entries = 1;
    char str[1024];
    size_t size;
    cl_uint tmp;
    cl_ulong utmp;
    cl_uint num_devices;
    
    // Device info.
    ret = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 1024, str, &size);
    if (ret == CL_SUCCESS) {
        std::cout << "Platform name      : " << str << std::endl;
    } else {
        printf("Error: getting platfotm info \n");
    }
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_entries, &device, &num_devices);
    cl_device_type t;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(t), &t, &size);
    if (t == CL_DEVICE_TYPE_CPU)
    {
        std::cout << "Device type        : CPU" << std::endl;
    }
    if (t == CL_DEVICE_TYPE_GPU)
    {
        std::cout << "Device type        : GPU" << std::endl;
    }
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(tmp), &tmp, &size);
    std::cout << "Number of units    : " << tmp << std::endl;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(utmp), &utmp, &size);
    std::cout << "Max. memory alloc. : " << utmp / (1024 * 1024) << " MB" << std::endl;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(utmp), &utmp, &size);
    std::cout << "Global mem. size   : " << utmp / (1024 * 1024) << " MB" << std::endl;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(tmp), &tmp, &size);
    std::cout << "Clock frequency    : " << tmp << " MHz " << std::endl;
    
    //----------------------------------------------------------------
    // creating context and Command Queue
    //----------------------------------------------------------------
    cl_context context;
    context = clCreateContext(NULL, num_devices, &device, NULL, NULL, &ret);
    
    CHECK_STATUS( ret, "Error: in Creating Context \n");
    
    // creating command queue
    cl_command_queue cq;
    
    cq = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
    
    CHECK_STATUS (ret , "Error: in Creating command Queue \n");
    
    cl_event event;
    
    //------------------------------------------------------------------------------
    // Load the kernel, creating the program, Build the program and create
    //-------------------------------------------------------------------------------
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen("mm_kernel.cl", "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load the kernel \n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose (fp);
    
    
    // creating a program with source
    cl_program program;
    //fprintf (stderr, "%s",source_str);
    program = clCreateProgramWithSource(context, 1, (const char **) &source_str,
                                        (const size_t *) &source_size, &ret);
    
    CHECK_STATUS(ret, "Error: in Creating The program \n");
    
    //Building the OpenCL program
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    CHECK_STATUS(ret,"Error: in Building The program \n");
    
    //creating the Kernel
    cl_kernel kernel;
    kernel = clCreateKernel(program, "matrix_mul", &ret);
    
    CHECK_STATUS(ret, "Error: in Creating The Kernel \n");
    
    //----------------------------------------------------------------------
    /* OpenCL buffers */
    //---------------------------------------------------------------------
    
    // creating the buffer in the HOST,
    size_t dim =  numData;
    size_t ldim = 8;
    const size_t num_elem = dim*dim; //size of element
    cl_int *A_host = (cl_int*)malloc(sizeof(cl_int) * num_elem);
    cl_int *B_host = (cl_int*)malloc(sizeof(cl_int) * num_elem);
    cl_int *C_host = (cl_int*)malloc(sizeof(cl_int) * num_elem);
    
    // initiating source buffer in host
    cl_int i ;
    for (i = 0; i< num_elem; i++)
    {
        A_host[i] = i;
        B_host[i] = i;
    }
    
    // allocating source buffer in GPU
    cl_mem A_device ;
    cl_mem B_device ;
    cl_mem C_device ;
    A_device = clCreateBuffer(context, CL_MEM_READ_ONLY,
                              num_elem*sizeof(cl_int), NULL, &ret);
    if (ret != CL_SUCCESS)    printf("Error: in allocating buffer A in GPU \n");
    
    B_device = clCreateBuffer(context, CL_MEM_READ_ONLY,
                              num_elem*sizeof(cl_int), NULL, &ret);
    if (ret != CL_SUCCESS)    printf("Error: in allocating buffer B in GPU \n");
    
    C_device =  clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                               num_elem*sizeof(cl_int), NULL, &ret);
    if (ret != CL_SUCCESS)    printf("Error: in allocating buffer C in GPU \n");
    
    // copy source buffer into GPU
    ret = clEnqueueWriteBuffer (cq, A_device, CL_TRUE, 0, num_elem *sizeof(cl_int),
                                A_host, 0, 0, &event );
    
    CHECK_STATUS(ret ,"Error: in copying source buffer into GPU \n");
    
    ret = clEnqueueWriteBuffer (cq, B_device, CL_TRUE, 0, num_elem *sizeof(cl_int),
                                B_host, 0, 0, &event );
    
    CHECK_STATUS(ret ,"Error: in copying source buffer into GPU \n");
    
    // setting the arguments
    ret = clSetKernelArg( kernel, 0, sizeof (cl_mem), &A_device); // 0 indicates the first argument
    if (ret != CL_SUCCESS)    printf("Error: setting the first argument \n");
    
    ret = clSetKernelArg( kernel, 1, sizeof (cl_mem), &B_device); // 1 indicates the second argument
    if (ret != CL_SUCCESS)    printf("Error: setting the second argument \n");
    
    ret = clSetKernelArg( kernel, 2, sizeof (cl_mem), &C_device); // 2 indicates the third argument
    if (ret != CL_SUCCESS)    printf("Error: setting the third argument \n");
    
    ret = clSetKernelArg( kernel, 3, sizeof (cl_int)* ldim*ldim, NULL); // 3 indicates the forth argument
    if (ret != CL_SUCCESS)    printf("Error: setting the forth argument \n");
    
    ret = clSetKernelArg( kernel, 4, sizeof (cl_int)* ldim*ldim, NULL); // 4 indicates the fifth argument
    if (ret != CL_SUCCESS)    printf("Error: setting the fifth argument \n");
    
    // main function for launching the kernel
    cl_uint dimension = 2;
    size_t global_work_size[2] = {dim, dim};
    size_t local_work_size[2] = {ldim, ldim};
    ret = clEnqueueNDRangeKernel (cq, kernel, dimension , NULL, global_work_size, local_work_size,
                                  0, NULL, &event);
    if (ret != CL_SUCCESS) {
        printf("Error: Launching Kernel \n");
    } else {
        // measure time
        cl_ulong            time_end;
        cl_ulong            time_start;
        ret = clWaitForEvents(1, &event);
        if (ret != CL_SUCCESS)    printf("Error: Waiting events \n");
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        std::cout << "Comuptation time                 : " << (time_end - time_start) / 1000 << " us" << std::endl;
    }
    
    // finish the execution
    ret = clFlush(cq);
    ret = clFinish(cq);
    
    if (ret != CL_SUCCESS)    printf("Error: Finishing the execution \n");
    
    //--------------------------------------------------------------------------
    /* Obtain the result from GPU to the CPU */
    //--------------------------------------------------------------------------
    
    // retrieving the buffer
    ret = clEnqueueReadBuffer (cq, C_device, CL_TRUE, 0, num_elem * sizeof(cl_int),
                               C_host, 0, NULL, &event);
    if (ret != CL_SUCCESS)    printf("Error: retrieving DST buffer into CPU \n");
    
    // Display the result to the screen
    
//    int j = 0;
//    for(i = 0; i < dim; i++)
//    {
//    for (j = 0; j < dim ; j++)
//    printf("A[%d.%d] = %d \t", A_host[i], A_host[j], A_host[i*dim+j]);
//    printf("\n");
//    }
//    printf("\n--------------------------------------------------------\n");
//    for(i = 0; i < dim; i++)
//    {
//    for (j = 0; j < dim ; j++)
//    printf("C[%d.%d] = %d \t", A_host[i], B_host[j], C_host[i*dim+j]);
//    printf("\n");
//    }
    
    fflush(NULL);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(A_device);
    ret = clReleaseMemObject(B_device);
    ret = clReleaseMemObject(C_device);
    ret = clReleaseCommandQueue(cq);
    ret = clReleaseContext(context);
    
    fflush(NULL);

    return(0);
}
