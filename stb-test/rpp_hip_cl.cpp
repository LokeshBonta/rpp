//==============================================================================
// Only compiled with hipcc compiler
//==============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <CL/cl.hpp>
//#include <rpp.h>
#include <rpp/rpp.h>
#include<rppi.h>
//#include<rppi_geometric_functions.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define CL_USE_DEPREviolet_3cED_OPENCL_1_2_APIS


template<typename T> // pointer type
void print3d_pkd(T matrix, size_t height, size_t width, size_t channel = 1)
{
    std::cout << std::endl;

    for(size_t i=0; i<height; i++ ){
        for(size_t j=0; j<width; j++ ){
            std::cout << "[";
            for(size_t k=0; k<channel; k++ ){
                std::cout << (unsigned int)( matrix[ k + j*channel + i*width*channel] );
                if (channel-1-k) std::cout << " ,";
            }
            std::cout << "]\t" ;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T> // pointer type
void print3d_pln(T matrix, size_t height, size_t width, size_t channel = 1)
{
    std::cout << std::endl;
    for(size_t k=0; k<channel; k++ ){
        std::cout << "[" << std::endl;
        for(size_t i=0; i<height; i++ ){
            for(size_t j=0; j<width; j++ ){
                std::cout << (unsigned int)( matrix[ j + i*width + k*width*height ] ) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }
}




int main( int argc, char* argv[] )
{
    typedef unsigned char TYPE_t;
    TYPE_t* h_a;
    TYPE_t* h_c;

    int height;
    int width;
    int channel;

    h_a = stbi_load( "/home/neel/ulagammai/sample_test/images/monarch.png",&width, &height, &channel, 0);

    size_t n = height * width * channel;
    size_t bytes = n*sizeof(TYPE_t);


    std::cout << "width:" << width << std::endl;
    std::cout << "height:" << height << std::endl;
    std::cout << "channel:" << channel << std::endl;

    

   
    h_c = (TYPE_t*)malloc(bytes);

//------ CL Alloc Stuffs
    cl_mem d_a;
    cl_platform_id platform_id;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_mem d_c;
    cl_context theContext;               // theContext
    cl_command_queue theQueue;           // command theQueue
    cl_program theProgram;               // theProgram
    cl_kernel theKernel;                 // theKernel

    cl_int err;

    hipStream_t stream;
    hipStreamCreate(&stream_)
    rppHandle_t handle;
    rppCreateWithStream(&handle, stream_);

    err = clGetPlatformIDs(1, &platform_id, NULL);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    theContext = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    //rppGetStream(handle, &theQueue);
    theQueue = clCreateCommandQueue(theContext, device_id, 0, &err);
    rppSetStream(handle, theQueue);

    d_a = clCreateBuffer(theContext, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    err = clEnqueueWriteBuffer(theQueue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);


    RppiSize srcSize;
    srcSize.height=height;
    srcSize.width=width;

    double alpha = 1;
    int beta = 50;


    rppi_brightness_u8_pkd3_gpu(d_a, srcSize, d_c,alpha,beta, handle);
    clEnqueueReadBuffer(theQueue, d_c, CL_TRUE, 0,
                             bytes, h_c, 0, NULL, NULL );


    stbi_write_png("./brightness_out.png",
                            width, height, channel, h_c, width *channel);

    
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_c);


    clReleaseCommandQueue(theQueue);
    clReleaseContext(theContext);

    free(h_a);
    free(h_c);

    return 0;
}
