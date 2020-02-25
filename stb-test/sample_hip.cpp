#include "hip/hip_runtime_api.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <rpp/rpp.h>
#include<rppi.h>
//#include<rppi_geometric_functions.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void check_hip_error(void)
{
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        {
            std::cerr
                << "Error: "
                << hipGetErrorString(err)
                << std::endl;
                exit(err);
        }
}


int main( int argc, char* argv[] )
{
    typedef unsigned char TYPE_t;
    TYPE_t* h_a;
    TYPE_t* h_b;


    int height;
    int width;
    int channel;

    h_a = stbi_load(argv[1],&width, &height, &channel, 0);

    size_t n = height * width * channel;
    h_b = (TYPE_t*)malloc(n * sizeof(unsigned char));
    long long size=sizeof(unsigned char)*n;
    int *in, *out;
    hipMalloc(&in,size);
    hipMalloc(&out,size);
    check_hip_error();

    hipMemcpy(in,h_a,n*sizeof(unsigned char),hipMemcpyHostToDevice);
    //hipDeviceSynchronize();
    check_hip_error();
    float alpha = 1.2;
    int beta =100;
    RppiSize srcSize;
    srcSize.height=height;
    srcSize.width=width;

    /* CALL RPP FUNCTION HERE*/
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStream(&handle, stream);
    rppi_brightness_u8_pkd3_gpu(in, srcSize, out,alpha, beta,handle);
    hipMemcpy(h_b,out,n*sizeof(unsigned char),hipMemcpyDeviceToHost);
    stbi_write_png("hip_out.png",
                            srcSize.width,srcSize.height, channel, h_b, srcSize.width *channel);


    hipFree(in);
    hipFree(out);
    check_hip_error();
    rppDestroy(handle);
    free(h_a);
    free(h_b);
}
