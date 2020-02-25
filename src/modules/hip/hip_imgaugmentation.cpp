#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"

// void max_size(Rpp32u *height, Rpp32u *width, unsigned int batch_size, unsigned int *max_height, unsigned int *max_width)
// {
//     int i;
//     *max_height  = 0;
//     *max_width =0;
//     for (i=0; i<batch_size; i++){
//         if(*max_height < height[i])
//             *max_height = height[i];
//         if(*max_width < width[i])
//             *max_width = width[i];
//     }
// }
/****************** Brightness ******************/

RppStatus
brightness_hip ( Rpp8u * srcPtr, RppiSize srcSize,
                            Rpp8u * dstPtr,
                            Rpp32f alpha, Rpp32s beta,
                            RppiChnFormat chnFormat, unsigned int channel,
                            rpp::Handle& handle) {

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "brightness_contrast.cpp", "brightness_contrast", vld, vgd, "")(srcPtr, dstPtr, alpha, beta, srcSize.height, srcSize.width, channel);
    return RPP_SUCCESS;

}

RppStatus brightness_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    // std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "brightness_contrast.cpp", "brightness_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        channel,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind
                                                                                        );
    return RPP_SUCCESS;
}

/***************** Contrast *********************/
RppStatus
contrast_hip (    Rpp8u * srcPtr, RppiSize srcSize,
                            Rpp8u * dstPtr,
                            Rpp32u newMin, Rpp32u newMax,
                            RppiChnFormat chnFormat, unsigned int channel,
                            rpp::Handle& handle)
{
    unsigned short counter=0;
    Rpp32u min = 0; /* Kernel has to be called */
    Rpp32u max = 255; /* Kernel has to be called */
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "contrast_stretch.cpp", "contrast_stretch", vld, vgd, "")(srcPtr, dstPtr, min, max, newMin, newMax, srcSize.height, srcSize.width, channel);
    return RPP_SUCCESS;

}



RppStatus
contrast_hip_batch (  Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;
    Rpp32u min = 0, max = 255;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    // std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "contrast.cpp", "contrast_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                min,
                                                                                max,
                                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                                handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                channel,
                                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                                plnpkdind
                                                                                );
    return RPP_SUCCESS;
}

/****************  Gamma correction *******************/
RppStatus
gamma_correction_hip ( Rpp8u * srcPtr,RppiSize srcSize,
                 Rpp8u * dstPtr, float gamma,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
         std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "gamma_correction.cpp", "gamma_correction", vld, vgd, "")(srcPtr, dstPtr, gamma, srcSize.height, srcSize.width, channel);
    return RPP_SUCCESS;

}

RppStatus
gamma_correction_hip_batch ( Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    // std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "gamma_correction.cpp", "gamma_correction_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        channel,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind
                                                                                        );
    return RPP_SUCCESS;
}
/********************** Exposure modification ************************/

RppStatus
exposure_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, Rpp32f exposureValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
         std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,  channel};
    handle.AddKernel("", "", "exposure.cpp", "exposure", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                exposureValue
                                                                );
    return RPP_SUCCESS;
}

RppStatus
exposure_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "exposure.cpp", "exposure_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}


/********************** Jitter ************************/
RppStatus
jitter_hip( Rpp8u * srcPtr,RppiSize srcSize, Rpp8u * dstPtr,
           unsigned int kernelSize,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    if(chnFormat == RPPI_CHN_PACKED)
    {
        handle.AddKernel("", "", "jitter.cpp", "jitter_pkd", vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     srcSize.height,
                                                                     srcSize.width,
                                                                     channel,
                                                                     kernelSize);
        // CreateProgramFromBinary(handle.GetStream(), "jitter.cpp", "jitter.cpp.bin", "jitter_pkd", theProgram, theKernel);
        // clRetainKernel(theKernel);
    }
    else if(chnFormat == RPPI_CHN_PLANAR)
    {
        handle.AddKernel("", "", "jitter.cpp", "jitter_pln", vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     srcSize.height,
                                                                     srcSize.width,
                                                                     channel,
                                                                     kernelSize);
        // CreateProgramFromBinary(handle.GetStream(), "jitter.cpp", "jitter.cpp.bin", "jitter_pln", theProgram, theKernel);
        // clRetainKernel(theKernel);
    }

    //---- Args Setter
    // err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &kernelSize);
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = 1;
    // cl_kernel_implementer (handle.GetStream(), gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}

RppStatus
jitter_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "jitter.cpp", "jitter_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/********************** Blend ************************/

RppStatus
blend_hip( Rpp8u* srcPtr1,Rpp8u* srcPtr2,
                 RppiSize srcSize, Rpp8u* dstPtr, float alpha,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "blend.cpp", "blend", vld, vgd, "")(srcPtr1,
                                                                srcPtr2,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                alpha,
                                                                channel);
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;
}
RppStatus
blend_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "blend.cpp", "blend_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
                                                                    handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                    channel,
                                                                    handle.GetInitHandle()->mem.mgpu.inc,
                                                                    plnpkdind
                                                                    );
    return RPP_SUCCESS;
}

//----
/********************** ADDING NOISE ************************/


RppStatus
noise_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                Rpp32f noiseProbability,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    srand(time(0));
    int ctr = 0;
    hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel,hipMemcpyDeviceToDevice);
    // clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
    if(noiseProbability != 0)
    {
        Rpp32u noisePixel = (Rpp32u)(noiseProbability * srcSize.width * srcSize.height );
        const Rpp32u pixelDistance = (srcSize.width * srcSize.height) / noisePixel;
        cl_kernel theKernel;
        cl_program theProgram;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
            handle.AddKernel("", "", "noise.cpp", "snp_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        pixelDistance
                                                                        );
            // CreateProgramFromBinary(handle, "noise.cpp", "noise.cpp.bin", "snp_pkd", theProgram, theKernel);
            // clRetainKernel(theKernel);
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, 1};
            handle.AddKernel("", "", "noise.cpp", "snp_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        pixelDistance
                                                                        );
            // CreateProgramFromBinary(handle, "noise.cpp", "noise.cpp.bin", "snp_pln", theProgram, theKernel);
            // clRetainKernel(theKernel);
        }
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &pixelDistance);

        // size_t gDim3[3];
        // gDim3[0] = srcSize.width;
        // gDim3[1] = srcSize.height;
        // gDim3[2] = 1;
        // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);
    }

    return RPP_SUCCESS;
}

RppStatus
noise_hip_batch (Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    Rpp32u nbatchSize = handle.GetBatchSize();
    hipMemcpy(srcPtr,dstPtr,sizeof(unsigned char) *
     (handle.GetInitHandle()->mem.mcpu.srcBatchIndex[nbatchSize-1] +
     (handle.GetInitHandle()->mem.mcpu.srcSize[nbatchSize-1].width *
     handle.GetInitHandle()->mem.mcpu.srcSize[nbatchSize-1].height) * channel),hipMemcpyDeviceToDevice);
    // clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) *
    //  (handle.GetInitHandle()->mem.mcpu.srcBatchIndex[nbatchSize-1] +
    //  (handle.GetInitHandle()->mem.mcpu.srcSize[nbatchSize-1].width *
    //  handle.GetInitHandle()->mem.mcpu.srcSize[nbatchSize-1].height) * channel), 0, NULL, NULL);
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "noise.cpp", "noise_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}


/********************** Rain ************************/

RppStatus
rain_hip(Rpp8u * srcPtr, RppiSize srcSize,Rpp8u * dstPtr, Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f transparency, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(rainPercentage == 0)
    {
        // clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
        hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel,hipMemcpyDeviceToDevice);
    }
    else
    {
        int ctr=0;
        Rpp32u rainDrops= (Rpp32u)((rainPercentage * srcSize.width * srcSize.height )/100);
        Rpp32u pixelDistance= (Rpp32u)((srcSize.width * srcSize.height) / rainDrops);
        transparency /= 5;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::cerr<<"\n Gonna call rain packed";
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width,srcSize.height,1};
            handle.AddKernel("", "", "rain.cpp", "rain_pkd", vld, vgd, "")(dstPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel,
                                                                    pixelDistance,
                                                                    rainWidth,
                                                                    rainHeight,
                                                                    transparency
                                                                    );
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width,srcSize.height,1};
            handle.AddKernel("", "", "rain.cpp", "rain_pln", vld, vgd, "")(dstPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel,
                                                                    pixelDistance,
                                                                    rainWidth,
                                                                    rainHeight,
                                                                    transparency
                                                                    );
        }

        // //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u *), &dstPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &pixelDistance);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &rainWidth);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &rainHeight);
        // clSetKernelArg(theKernel, ctr++, sizeof(float), &transparency);
        // //----

        // size_t gDim3[3];
        // gDim3[0] = srcSize.width;
        // gDim3[1] = srcSize.height;
        // gDim3[2] = 1;
        // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);

        // cl_kernel theKernel1;
        // cl_program theProgram1;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height,channel};
        std::cerr<<"\n Gonna call rain\n";
        handle.AddKernel("", "", "rain.cpp", "rain", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel
                                                                );
        // CreateProgramFromBinary(handle,"rain.cpp","rain.cpp.bin","rain",theProgram1,theKernel1);
        // clRetainKernel(theKernel);
        // ctr=0;
        // clSetKernelArg(theKernel1, ctr++, sizeof(Rpp8u *), &srcPtr);
        // clSetKernelArg(theKernel1, ctr++, sizeof(Rpp8u *), &dstPtr);
        // clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &channel);
        // gDim3[2] = channel;
        // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram1, theKernel1);
    }
    std::cerr<<"\n Gonna return Success";
    return RPP_SUCCESS;
}
RppStatus
rain_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    Rpp32u nbatchSize = handle.GetBatchSize();
    hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) *
     (handle.GetInitHandle()->mem.mcpu.srcBatchIndex[nbatchSize-1] +
     (handle.GetInitHandle()->mem.mgpu.csrcSize.width[nbatchSize-1] *
     handle.GetInitHandle()->mem.mgpu.csrcSize.height[nbatchSize-1]) * channel),hipMemcpyDeviceToDevice);
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "rain.cpp", "rain_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}
/********************** Snow ************************/
RppStatus
snow_hip( Rpp8u * srcPtr,RppiSize srcSize, Rpp8u * dstPtr,
           float snowCoefficient,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle)
{
    if(snowCoefficient == 0)
    {
        hipMemcpy(dstPtr, srcPtr,sizeof(unsigned char) * srcSize.width * srcSize.height * channel, hipMemcpyDeviceToDevice);
    }
    else
    {
        int ctr=0; 
        Rpp32u snowDrops= (Rpp32u)((snowCoefficient * srcSize.width * srcSize.height )/100);
        Rpp32u pixelDistance= (Rpp32u)((srcSize.width * srcSize.height) / snowDrops);
        size_t gDim3[3];
        gDim3[0] = srcSize.width;
        gDim3[1] = srcSize.height;
        gDim3[2] = 1;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "snow.cpp", "snow_pkd", vld, vgd, "")(
                                                                    dstPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel,
                                                                    pixelDistance
                                                                    );
            // CreateProgramFromBinary(handle.GetStream(),"snow.cpp","snow.cpp.bin","snow_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "snow.cpp", "snow_pln", vld, vgd, "")(
                                                                    dstPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel,
                                                                    pixelDistance
                                                                    );
            // CreateProgramFromBinary(handle.GetStream(),"snow.cpp","snow.cpp.bin","snow_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }

        //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(cl_mem), &dstPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &pixelDistance);
        // cl_kernel_implementer (handle.GetStream(), gDim3, NULL/*Local*/, theProgram, theKernel);
        
        gDim3[0] = srcSize.width;
        gDim3[1] = srcSize.height;
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "snow.cpp", "snow", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel);
        // CreateProgramFromBinary(handle.GetStream(),"snow.cpp","snow.cpp.bin","snow",theProgram1,theKernel1);
        // clSetKernelArg(theKernel1, ctr++, sizeof(cl_mem), &srcPtr);
        // clSetKernelArg(theKernel1, ctr++, sizeof(cl_mem), &dstPtr);
        // clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel1, ctr++, sizeof(unsigned int), &channel);
        // cl_kernel_implementer (handle.GetStream(), gDim3, NULL/*Local*/, theProgram1, theKernel1);
    }

    return RPP_SUCCESS;
}

RppStatus
snow_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    Rpp32u nbatchSize = handle.GetBatchSize();
    // clEnqueueCopyBuffer(handle.GetStream(), srcPtr, dstPtr, 0, 0, sizeof(unsigned char) *
    //  (handle.GetInitHandle()->mem.mcpu.srcBatchIndex[nbatchSize-1] +
    //  (handle.GetInitHandle()->mem.mcpu.srcSize[nbatchSize-1].width *
    //  handle.GetInitHandle()->mem.mcpu.srcSize[nbatchSize-1].height) * channel), 0, NULL, NULL);
    hipMemcpy(dstPtr,srcPtr,sizeof(unsigned char) *
     (handle.GetInitHandle()->mem.mcpu.srcBatchIndex[nbatchSize-1] +
     (handle.GetInitHandle()->mem.mgpu.csrcSize.width[nbatchSize-1] *
     handle.GetInitHandle()->mem.mgpu.csrcSize.height[nbatchSize-1]) * channel),hipMemcpyDeviceToDevice);
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "snow.cpp", "snow_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}
/********************** Fog ************************/

RppStatus
fog_hip( Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * temp, Rpp32f fogValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if(fogValue == 0)
    {
        // clEnqueueCopyBuffer(handle.GetStream(), temp, srcPtr, 0, 0, sizeof(unsigned char) * srcSize.width * srcSize.height * channel, 0, NULL, NULL);
        hipMemcpy(srcPtr,temp,sizeof(unsigned char) * srcSize.width * srcSize.height * channel,hipMemcpyDeviceToDevice);

    }
    else
    {
        int ctr=0;
        cl_kernel theKernel;
        cl_program theProgram;

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};
            handle.AddKernel("", "", "fog.cpp", "fog_planar", vld, vgd, "")(srcPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                fogValue
                                                                );
            // CreateProgramFromBinary(handle,"fog.cpp","fog.cpp.bin","fog_planar",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};
            handle.AddKernel("", "", "fog.cpp", "fog_pkd", vld, vgd, "")(srcPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                fogValue
                                                                );
            // CreateProgramFromBinary(handle,"fog.cpp","fog.cpp.bin","fog_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}
        //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u *), &srcPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, ctr++, sizeof(float), &fogValue);
        // //----

        // size_t gDim3[3];
        // gDim3[0] = srcSize.width;
        // gDim3[1] = srcSize.height;
        // gDim3[2] = 1;
        // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);
    }

    return RPP_SUCCESS;
}
RppStatus
fog_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "fog.cpp", "fog_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/********************** pixelate ************************/

RppStatus
pixelate_hip(Rpp8u * srcPtr, RppiSize srcSize,Rpp8u * dstPtr,
            RppiChnFormat chnFormat,
            unsigned int channel,rpp::Handle& handle)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;

    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width / 7) + 1,(srcSize.height / 7) + 1, channel};
        handle.AddKernel("", "", "pixelate.cpp", "pixelate_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    else if(chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width / 7) + 1,(srcSize.height / 7) + 1, channel};
        handle.AddKernel("", "", "pixelate.cpp", "pixelate_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }

    // //---- Args Setter
    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u *), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u *), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // size_t gDim3[3];
    // gDim3[0] = ceil(srcSize.width / 7) + 1;
    // gDim3[1] = ceil(srcSize.height / 7) + 1;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}

RppStatus
pixelate_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{

    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};
    handle.AddKernel("", "", "pixelate.cpp", "pixelate_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

// /********************** Random Shadow ************************/
RppStatus
random_shadow_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u x1, Rpp32u y1,
                 Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX,
                 Rpp32u maxSizeY, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32u row1, row2, column2, column1;
    int i, j, ctr = 0;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
    handle.AddKernel("", "", "random_shadow.cpp","random_shadow", vld, vgd, "")(srcPtr, dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);


    for(i = 0 ; i < numberOfShadows ; i++)
    {
        ctr = 0;
        do
        {
            row1 = rand() % srcSize.height;
            column1 = rand() % srcSize.width;
        } while (column1 <= x1 || column1 >= x2 || row1 <= y1 || row1 >= y2);
        do
        {
            row2 = rand() % srcSize.height;
            column2 = rand() % srcSize.width;
        } while ((row2 < row1 || column2 < column1) || (column2 <= x1 || column2 >= x2 || row2 <= y1 || row2 >= y2) || (row2 - row1 >= maxSizeY || column2 - column1 >= maxSizeX));

        if(RPPI_CHN_PACKED == chnFormat)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
            handle.AddKernel("", "", "random_shadow.cpp", "random_shadow_packed", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                column1,
                                                                row1,
                                                                column2,
                                                                row2
                                                                );
            // CreateProgramFromBinary(handle, "random_shadow.cpp", "random_shadow.cpp.bin", "random_shadow_packed", theProgram, theKernel);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
            handle.AddKernel("", "", "random_shadow.cpp", "random_shadow_planar", vld, vgd, "")(srcPtr,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel,
                                                                column1,
                                                                row1,
                                                                column2,
                                                                row2
                                                                );
            // cl_kernel_initializer(handle.GetStream(),
            //                     "random_shadow.cpp",
            //                     "random_shadow_planar",
            //                     theProgram, theKernel);
        }
        //---- Args Setter
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &column1);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &row1);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &column2);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &row2);
        // size_t gDim3[3];
        // gDim3[0] = column2 - column1;
        // gDim3[1] = row2 - row1;
        // gDim3[2] = channel;
        // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);
    }
    return RPP_SUCCESS;
}

RppStatus
random_shadow_hip_batch(   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
//     Rpp8u* srcPtr, RppiSize *srcSize, Rpp8u* dstPtr,
//  Rpp32u *x1, Rpp32u *y1, Rpp32u *x2, Rpp32u *y2, Rpp32u *numberOfShadows, Rpp32u *maxSizeX, Rpp32u *maxSizeY, Rpp32u nBatchSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle,)
{
    // cl_context theContext;
    // clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &NULL);
    // cl_device_id theDevice;            
    // clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    unsigned int maxHeight, maxWidth, maxKernelSize;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }
    // Rpp8u* srcPtr1 = hipMalloc(Rpp8u*_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    // Rpp8u* dstPtr1 = hipMalloc(Rpp8u*_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    Rpp8u *srcPtr1, *dstPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    hipMalloc(&dstPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    size_t batchIndex = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        Rpp32u row1, row2, column2, column1;
        int x, y;

        // clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, 
        // sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        // clEnqueueCopyBuffer(handle.GetStream(), srcPtr1, dstPtr1, 0, 0,
        //  sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        hipMemcpy(srcPtr1, srcPtr+batchIndex , sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        hipMemcpy(dstPtr1, srcPtr1,  sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        for(x = 0 ; x < handle.GetInitHandle()->mem.mcpu.uintArr[4].uintmem[i]; x++)
        {
            do
            {
                row1 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
                column1 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            } while (column1 <= handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] || column1 >= handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i] || row1 <= handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] || row1 >= handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i]);
            do
            {
                row2 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
                column2 = rand() % handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            } while ((row2 < row1 || column2 < column1) || (column2 <= handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i] || column2 >= handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i]
            || row2 <= handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i] || row2 >= handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i]) || (row2 - row1 >= handle.GetInitHandle()->mem.mcpu.uintArr[6].uintmem[i]
            || column2 - column1 >= handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i]));

            if(RPPI_CHN_PACKED == chnFormat)
            {
                std::vector<size_t> vld{32, 32, 1};
                std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
                handle.AddKernel("", "", "random_shadow.cpp", "random_shadow_packed", vld, vgd, "")(srcPtr1,
                                                                                                   dstPtr1,
                                                                                                   handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                   handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                   channel,
                                                                                                   column1,row1,
                                                                                                   column2,row2);
                // CreateProgramFromBinary(handle.GetStream(), "random_shadow.cpp", "random_shadow.cpp.bin", "random_shadow_packed", theProgram, theKernel);
                // clRetainKernel(theKernel);
            }
            else
            {
                std::vector<size_t> vld{32, 32, 1};
                std::vector<size_t> vgd{column2 - column1,row2 - row1,channel};
                handle.AddKernel("", "", "random_shadow.cpp", "random_shadow_planar", vld, vgd, "")(srcPtr1,
                                                                                                   dstPtr1,
                                                                                                   handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                                   handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                                   channel,
                                                                                                   column1,row1,
                                                                                                   column2,row2);
                    // CreateProgramFromBinary(handle.GetStream(), "random_shadow.cpp", "random_shadow.cpp.bin", "random_shadow_planar", theProgram, theKernel);
                    // clRetainKernel(theKernel);
            }
            //---- Args Setter
            // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr1);
            // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr1);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &column1);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &row1);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &column2);
            // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &row2);
            // size_t gDim3[3];
            // gDim3[0] = column2 - column1;
            // gDim3[1] = row2 - row1;
            // gDim3[2] = channel;
            // cl_kernel_implementer (handle.GetStream(), gDim3, NULL/*Local*/, theProgram, theKernel);
        }
        // clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndex, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        hipMemcpy(dstPtr+batchIndex, dstPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
    }
    return RPP_SUCCESS;
}

// /********************** Histogram balance ************************/
RppStatus
histogram_balance_hip(Rpp8u* srcPtr, RppiSize srcSize,
                Rpp8u* dstPtr,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    unsigned int numGroups;

    size_t lDim3[3];
    size_t gDim3[3];
    int num_pixels_per_work_item = 16;

    gDim3[0] = srcSize.width / num_pixels_per_work_item ;// Plus 1
    gDim3[1] = srcSize.height / num_pixels_per_work_item ;
    lDim3[0] = num_pixels_per_work_item;
    lDim3[1] = num_pixels_per_work_item;
    gDim3[2] = 1;
    lDim3[2] = 1;
    

    numGroups = gDim3[0] * gDim3[1];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;
    
    Rpp8u* partialHistogram;
    hipMalloc(&partialHistogram,sizeof(unsigned int)*256*numGroups);
    Rpp8u* histogram;
    hipMalloc(&histogram,sizeof(unsigned int)*256);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "histogram.cpp", "partial_histogram_pln", vld, vgd, "")(srcPtr,
                                                                                        partialHistogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
        // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin","partial_histogram_pln",
        //                             theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "histogram.cpp", "partial_histogram_pkd", vld, vgd, "")(srcPtr,
                                                                                        partialHistogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
        // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin","partial_histogram_pkd",
        //                             theProgram,theKernel);
        // clRetainKernel(theKernel);

    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    // counter = 0;
    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &partialHistogram);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);


    // cl_kernel_implementer (handle.GetStream(), gDim3, lDim3, theProgram, theKernel);

    // // For sum histogram kernel
    gDim3[0] = 256;
    lDim3[0] = 256;
    gDim3[1] = 1; 
    gDim3[2] = 1;
    lDim3[1] = 1;
    lDim3[2] = 1;
    std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
    handle.AddKernel("", "", "histogram.cpp", "histogram_sum_partial", vld, vgd, "")(partialHistogram,
                                                                                    histogram,
                                                                                    numGroups);
    // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin","histogram_sum_partial",
    //                                                                     theProgram,theKernel);
    // clRetainKernel(theKernel);

    // counter = 0;
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &partialHistogram);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &histogram);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &numGroups);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // cl_kernel_implementer (handle.GetStream(), gDim3, lDim3, theProgram, theKernel);
   
    Rpp8u* cum_histogram;
    hipMalloc(&cum_histogram,sizeof(unsigned int)*256);
    // For scan kernel
    gDim3[0] = 256;
    gDim3[1] = 1; 
    gDim3[2] = 1;
    lDim3[0] = 32;
    lDim3[1] = 1; 
    lDim3[2] = 1;
    std::vector<size_t> vld1{lDim3[0], lDim3[1], lDim3[2]};
    std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};
    handle.AddKernel("", "", "scan.cpp", "scan", vld1, vgd1, "")(histogram,
                                                             cum_histogram);
    // CreateProgramFromBinary(handle.GetStream(),"scan.cpp","scan.cpp.bin","scan",theProgram,theKernel);
    // clRetainKernel(theKernel);

    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &histogram);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &cum_histogram);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), NULL);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), NULL);



    // cl_kernel_implementer (handle.GetStream(), gDim3, lDim3, theProgram, theKernel);

    // For histogram equalize

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "histogram.cpp", "histogram_equalize_pln", vld, vgd, "")(srcPtr,
                                                                                        dstPtr,
                                                                                        cum_histogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
        // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin",
        //                             "histogram_equalize_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "histogram.cpp", "histogram_equalize_pkd", vld, vgd, "")(srcPtr,
                                                                                        dstPtr,
                                                                                        cum_histogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
        // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin","histogram_equalize_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}
    // counter = 0;
    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &cum_histogram);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);

    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // //std::cerr<<"Hist equalize call"<<std::endl;
    // cl_kernel_implementer (handle.GetStream(), gDim3, NULL, theProgram, theKernel);
    //std::cerr<<"Hist equalize return"<<std::endl;
    hipFree(cum_histogram);
    hipFree(partialHistogram);
    hipFree(histogram);
    //Freeing the Memory is yet to be done!
    return RPP_SUCCESS;
}

RppStatus
histogram_balance_hip_batch (Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
    // Rpp8u* srcPtr, RppiSize *srcSize, RppiSize *maxsrcSize, 
    //                         Rpp8u* dstPtr,RppiROI *roiPoints,   Rpp32u nBatchSize,
    //                         RppiChnFormat chnFormat, unsigned int channel,
    //                         rpp::Handle& handle,){
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    int numGroups = 0;
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        int size = 0;
        size = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel;
        int group = std::ceil(size / 256);
        if(numGroups < group)
            numGroups = group;
    }

    Rpp8u* partialHistogram;
    hipMalloc(&partialHistogram,sizeof(unsigned int)*256*numGroups);
    Rpp8u* histogram;
    hipMalloc(&histogram,sizeof(unsigned int)*256);
    Rpp8u* cum_histogram;
    hipMalloc(&cum_histogram,sizeof(unsigned int)*256);
    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1,sizeof(unsigned char) * maxHeight * maxWidth * channel);
    Rpp8u* dstPtr1;
    hipMalloc(&dstPtr1,sizeof(unsigned char)* maxHeight * maxWidth * channel);

    int ctr;
           
    size_t gDim3[3];

    size_t batchIndex = 0;
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        size_t lDim3[3];
        size_t gDim3[3];
        int num_pixels_per_work_item = 16;

        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] / num_pixels_per_work_item ;
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] / num_pixels_per_work_item ;
        lDim3[0] = num_pixels_per_work_item;
        lDim3[1] = num_pixels_per_work_item;
        gDim3[2] = 1;
        lDim3[2] = 1;
        

        numGroups = gDim3[0] * gDim3[1];
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        hipMemcpy(srcPtr1, srcPtr+batchIndex , sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "histogram.cpp", "partial_histogram_pln", vld, vgd, "")(srcPtr,
                                                                                            partialHistogram,
                                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin","partial_histogram_pln",
            //                             theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
            std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
            handle.AddKernel("", "", "histogram.cpp", "partial_histogram_pkd", vld, vgd, "")(srcPtr,
                                                                                            partialHistogram,
                                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                            handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                            channel);
            // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin","partial_histogram_pkd",
            //                             theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}

        // counter = 0;
        // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr1);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &partialHistogram);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);


        // cl_kernel_implementer (handle.GetStream(), gDim3, lDim3, theProgram, theKernel);

        // // For sum histogram kernel
        gDim3[0] = 256;
        lDim3[0] = 256;
        gDim3[1] = 1; 
        gDim3[2] = 1;
        lDim3[1] = 1;
        lDim3[2] = 1;
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "histogram.cpp", "histogram_sum_partial", vld, vgd, "")(partialHistogram,
                                                                                        histogram,
                                                                                        numGroups);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &partialHistogram);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &histogram);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &numGroups);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);

        // For scan kernel
        // counter = 0;
        gDim3[0] = 256;
        gDim3[1] = 1; 
        gDim3[2] = 1;
        lDim3[0] = 32;
        lDim3[1] = 1; 
        lDim3[2] = 1;
        std::vector<size_t> vld1{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "scan.cpp", "scan", vld1, vgd1, "")(histogram,
                                                             cum_histogram);
        // CreateProgramFromBinary(handle.GetStream(),"scan.cpp","scan.cpp.bin","scan",theProgram,theKernel);
        // clRetainKernel(theKernel);

        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &histogram);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &cum_histogram);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), NULL);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), NULL);

        // gDim3[0] = 256;
        // gDim3[1] = 1; 
        // gDim3[2] = 1;
        // lDim3[0] = 32;
        // lDim3[1] = 1; 
        // lDim3[2] = 1;
        // cl_kernel_implementer (handle.GetStream(), gDim3, lDim3, theProgram, theKernel);

        // For histogram equalize

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],channel};
            handle.AddKernel("", "", "histogram.cpp", "histogram_equalize_pln", vld, vgd, "")(srcPtr1,
                                                                                        dstPtr1,
                                                                                        cum_histogram,
                                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                        channel);
            // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin",
            //                             "histogram_equalize_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],channel};
            handle.AddKernel("", "", "histogram.cpp", "histogram_equalize_pkd", vld, vgd, "")(srcPtr1,
                                                                                        dstPtr1,
                                                                                        cum_histogram,
                                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                        channel);
            // CreateProgramFromBinary(handle.GetStream(),"histogram.cpp","histogram.cpp.bin",
            // "histogram_equalize_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}
        // counter = 0;
        // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr1);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr1);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &cum_histogram);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);

        // gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        // gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        // gDim3[2] = channel;
        // //std::cerr<<"Hist equalize call"<<std::endl;
        // cl_kernel_implementer (handle.GetStream(), gDim3, NULL, theProgram, theKernel);
        hipMemcpy(dstPtr+batchIndex, dstPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
    }

    hipFree(cum_histogram);
    hipFree(partialHistogram);
    hipFree(histogram);
    return RPP_SUCCESS;
}
/********************** Occlusion ************************/
RppStatus
occlusion_hip(   Rpp8u* srcPtr1,RppiSize srcSize1,
                Rpp8u* srcPtr2,RppiSize srcSize2, Rpp8u* dstPtr,//Destiation Size is Same as the Second Image's Dimensions
                const unsigned int x11,
                const unsigned int y11,
                const unsigned int x12,
                const unsigned int y12,
                const unsigned int x21,
                const unsigned int y21,
                const unsigned int x22,
                const unsigned int y22,
                RppiChnFormat chnFormat,unsigned int channel,
                rpp::Handle& handle)
{
    size_t gDim3[3];
    gDim3[0] = srcSize2.width;
    gDim3[1] = srcSize2.height;
    gDim3[2] = channel;
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
     if (chnFormat == RPPI_CHN_PLANAR)
    {
        handle.AddKernel("", "", "occlusion.cpp", "occlusion_pln", vld, vgd, "")(srcPtr1,
                                                                                srcPtr2,
                                                                                dstPtr,
                                                                                srcSize1.height,
                                                                                srcSize1.width,
                                                                                srcSize2.height,
                                                                                srcSize2.width,
                                                                                x11,
                                                                                y11,
                                                                                x12,
                                                                                y12,
                                                                                x21,
                                                                                y21,
                                                                                x22,
                                                                                y22,
                                                                                channel);
        // CreateProgramFromBinary(handle.GetStream(),"occlusion.cpp","occlusion.cpp.bin","occlusion_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
            handle.AddKernel("", "", "occlusion.cpp", "occlusion_pkd", vld, vgd, "")(srcPtr1,
                                                                                srcPtr2,
                                                                                dstPtr,
                                                                                srcSize1.height,
                                                                                srcSize1.width,
                                                                                srcSize2.height,
                                                                                srcSize2.width,
                                                                                x11,
                                                                                y11,
                                                                                x12,
                                                                                y12,
                                                                                x21,
                                                                                y21,
                                                                                x22,
                                                                                y22,
                                                                                channel);
        // CreateProgramFromBinary(handle.GetStream(),"occlusion.cpp","occlusion.cpp.bin","occlusion_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {std::cerr << "Internal error: Unknown Channel format";}

    // //---- Args Setter
    // int ctr =0;
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr1);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr2);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize1.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize2.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize2.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x11);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y11);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x12);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y12);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x21);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y21);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x22);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y22);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // //----

    return RPP_SUCCESS;
}

RppStatus
occlusion_hip_batch (Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    unsigned int src1MaxHeight, src1MaxWidth,src2MaxHeight, src2MaxWidth, dstMaxHeight, dstMaxWidth;
    src1MaxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    src1MaxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(src1MaxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            src1MaxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(src1MaxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            src1MaxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        if(src2MaxHeight < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
            src2MaxHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        if(src2MaxWidth < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
            src2MaxWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
    }

    Rpp8u* srcPtr11;
    hipMalloc(&srcPtr11, sizeof(unsigned char) * src1MaxHeight * src2MaxWidth * channel);
    Rpp8u* srcPtr21;
    hipMalloc(&srcPtr21, sizeof(unsigned char) * src2MaxHeight * src2MaxWidth * channel);
    Rpp8u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned char) * src2MaxHeight * src2MaxWidth * channel);

    int ctr;
    size_t gDim3[3];
    size_t src1BatchIndex = 0, src2BatchIndex =0;
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        size_t gDim3[3];
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        hipMemcpy(srcPtr11, srcPtr1+src1BatchIndex, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel,hipMemcpyDeviceToDevice);
        hipMemcpy(srcPtr21, srcPtr2+src2BatchIndex, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel,hipMemcpyDeviceToDevice);
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            handle.AddKernel("", "", "occlusion.cpp", "occlusion_pln", vld, vgd, "")(srcPtr11,
                                                                                srcPtr21,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[4].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[6].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[7].uintmem[i],
                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"occlusion.cpp","occlusion.cpp.bin","occlusion_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "occlusion.cpp", "occlusion_pkd", vld, vgd, "")(srcPtr11,
                                                                                srcPtr21,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[0].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[3].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[4].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[6].uintmem[i],
                                                                                handle.GetInitHandle()->mem.mcpu.uintArr[7].uintmem[i],
                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"occlusion.cpp","occlusion.cpp.bin","occlusion_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}

        //---- Args Setter
        // int ctr =0;
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr11);
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr21);
        // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr1);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.height[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.width[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x11[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y11[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x12[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y12[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x21[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y21[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &x22[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &y22[i]);
        // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // cl_kernel_implementer (handle.GetStream(), gDim3, NULL/*Local*/, theProgram, theKernel);

        hipMemcpy( dstPtr+src2BatchIndex, dstPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * channel, hipMemcpyDeviceToDevice);
        src1BatchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        src2BatchIndex += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);
        //sBatchIndex += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);
    }
   /* Releasing of the stuff needs to be done */
    hipFree(srcPtr11);
    hipFree(srcPtr21);
    hipFree(dstPtr1);
    return RPP_SUCCESS;   
}
