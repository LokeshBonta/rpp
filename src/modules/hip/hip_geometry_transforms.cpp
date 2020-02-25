#include "hip_declarations.hpp"

/************* Fish eye ******************/
RppStatus
fisheye_hip(Rpp8u* srcPtr, RppiSize srcSize,
                Rpp8u* dstPtr,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "fish_eye.cpp", "fisheye_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "fish_eye.cpp", "fisheye_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    // else
    // {std::cerr << "Internal error: Unknown Channel format";}
    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;

}

RppStatus
fisheye_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "fish_eye.cpp", "fisheye_batch", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                            handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                            handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                            handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                            handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                            handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                            handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                            handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                            channel,
                                                                            handle.GetInitHandle()->mem.mgpu.inc,
                                                                            plnpkdind);
    return RPP_SUCCESS;
}

/************* LensCorrection ******************/
RppStatus
lens_correction_hip( Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr,
           float strength,float zoom,
           RppiChnFormat chnFormat, unsigned int channel,
           rpp::Handle& handle)
{
    unsigned short counter=0;
    if (strength == 0)
        strength = 0.000001;
    float halfWidth = (float)srcSize.width / 2.0;
    float halfHeight = (float)srcSize.height / 2.0;
    float correctionRadius = (float)sqrt((float)srcSize.width * srcSize.width + srcSize.height * srcSize.height) / (float)strength;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "lens_correction.cpp", "lenscorrection_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        strength,
                                                                        zoom,
                                                                        halfWidth,
                                                                        halfHeight,
                                                                        correctionRadius,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
        handle.AddKernel("", "", "lens_correction.cpp", "lenscorrection_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        strength,
                                                                        zoom,
                                                                        halfWidth,
                                                                        halfHeight,
                                                                        correctionRadius,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }
    // //---- Args Setter
    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &strength);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &zoom);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &halfWidth);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &halfHeight);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &correctionRadius);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // //----

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}

RppStatus
lens_correction_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }

    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    Rpp8u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);

    size_t gDim3[3];
    size_t batchIndex = 0;

    for(int i = 0 ; i < handle.GetBatchSize() ; i++)
    {
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        gDim3[2] = channel;
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
        if (handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i] == 0)
                handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i] = 0.000001;
        float halfWidth = (float)handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] / 2.0;
        float halfHeight = (float)handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] / 2.0;
        float correctionRadius = (float)sqrt((float)handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] + handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] 
                                    * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]) / (float)handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i];
        hipMemcpy(srcPtr1, srcPtr+batchIndex , sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            handle.AddKernel("", "", "lens_correction.cpp", "lenscorrection_pln", vld, vgd, "")(srcPtr1,
                                                                                               dstPtr1,
                                                                                               handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i],
                                                                                               handle.GetInitHandle()->mem.mcpu.floatArr[1].floatmem[i],
                                                                                               halfWidth,
                                                                                               halfHeight,
                                                                                               correctionRadius,
                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                               channel);
            // CreateProgramFromBinary(handle.GetStream(),"lens_correction.cpp","lens_correction.bin","lenscorrection_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            handle.AddKernel("", "", "lens_correction.cpp", "lenscorrection_pkd", vld, vgd, "")(srcPtr1,
                                                                                               dstPtr1,
                                                                                               handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i],
                                                                                               handle.GetInitHandle()->mem.mcpu.floatArr[1].floatmem[i],
                                                                                               halfWidth,
                                                                                               halfHeight,
                                                                                               correctionRadius,
                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                               handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                               channel);
            // CreateProgramFromBinary(handle.GetStream(),"lens_correction.cpp","lens_correction.bin","lenscorrection_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {std::cerr << "Internal error: Unknown Channel format";}
        
        // //---- Args Setter
        // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr1);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr1);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &handle.GetInitHandle()->mem.mcpu.floatArr[0].floatmem[i]);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &handle.GetInitHandle()->mem.mcpu.floatArr[1].floatmem[i]);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &halfWidth);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &halfHeight);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(float), &correctionRadius);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
        // //----
        // cl_kernel_implementer (handle.GetStream(), gDim3, NULL/*Local*/, theProgram, theKernel);

        hipMemcpy(dstPtr+batchIndex, dstPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
    }
    return RPP_SUCCESS; 
}

/************* Flip ******************/

RppStatus
flip_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, uint flipAxis,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)

{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (flipAxis == 1)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_vertical_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
        else if (flipAxis == 0)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_horizontal_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
        else if (flipAxis == 2)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_bothaxis_planar", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (flipAxis == 1)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_vertical_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
        else if (flipAxis == 0)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_horizontal_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
        else if (flipAxis == 2)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};
            handle.AddKernel("", "", "flip.cpp", "flip_bothaxis_packed", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel
                                                                        );
        }
    }
     return RPP_SUCCESS;
}

RppStatus
flip_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
   // std::cout << "coming till near kernel here" << std::endl;
    handle.AddKernel("", "", "flip.cpp", "flip_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,   
                                                                        channel,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        plnpkdind);


    return RPP_SUCCESS;
}

/************* Resize ******************/
RppStatus
resize_hip(Rpp8u * srcPtr, RppiSize srcSize,
                Rpp8u * dstPtr, RppiSize dstSize,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }

    // size_t gDim3[3];
    // gDim3[0] = dstSize.width;
    // gDim3[1] = dstSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;
}

RppStatus
resize_hip_batch (   Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

        //unsigned int padding = 0;
        //unsigned int type = 0;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "resize.cpp", "resize_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/************* Resize Crop ******************/
RppStatus
resize_crop_hip(Rpp8u * srcPtr, RppiSize srcSize,
                Rpp8u * dstPtr, RppiSize dstSize,
                Rpp32u x1, Rpp32u x2, Rpp32u y1, Rpp32u y2, 
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    // unsigned short counter=0;
    // cl_int err;
    // cl_kernel theKernel;
    // cl_program theProgram;
    // cl_context theContext;
    unsigned int type = 0,  padding = 0;
    unsigned int width,height;
    if(type == 1)
    {
        width = dstSize.width - padding * 2;
        height = dstSize.height - padding * 2;
    }
    else
    {
        width = dstSize.width;
        height = dstSize.height;
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.height);
        // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.width);
    }
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{width, height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_crop_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        height,
                                                                        width,
                                                                        x1,
                                                                        x2,
                                                                        y1,
                                                                        y2,
                                                                        padding,
                                                                        type,
                                                                        channel
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{width, height, channel};
        handle.AddKernel("", "", "resize.cpp", "resize_crop_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        height,
                                                                        width,
                                                                        x1,
                                                                        x2,
                                                                        y1,
                                                                        y2,
                                                                        padding,
                                                                        type,
                                                                        channel
                                                                        );
    }

    // if (chnFormat == RPPI_CHN_PLANAR)
    // {
    //     CreateProgramFromBinary(handle,"resize.cpp","resize.cpp.bin","resize_crop_pln",theProgram,theKernel);
    //     clRetainKernel(theKernel);
    //     // cl_kernel_initializer(  "resize.cpp", "resize_crop_pln",
    //     //                         theProgram, theKernel);
    // }
    // else if (chnFormat == RPPI_CHN_PACKED)
    // {
    //     CreateProgramFromBinary(handle,"resize.cpp","resize.cpp.bin","resize_crop_pkd",theProgram,theKernel);
    //     clRetainKernel(theKernel);
    //     // cl_kernel_initializer(  "resize.cpp", "resize_crop_pkd",
    //     //                         theProgram, theKernel);
    // }
    // else
    // {std::cerr << "Internal error: Unknown Channel format";}
    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &x1);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &y1);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &x2);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &y2);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &padding);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &type);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);


    // size_t gDim3[3];
    // if(type == 1)
    // {
    //     gDim3[0] = dstSize.width - padding * 2;
    //     gDim3[1] = dstSize.height - padding * 2;
    //     gDim3[2] = channel;
    // }
    // else
    // {
    //     gDim3[0] = dstSize.width;
    //     gDim3[1] = dstSize.height;
    //     gDim3[2] = channel;
    // }

    // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}
RppStatus
resize_crop_hip_batch (   Rpp8u * srcPtr, Rpp8u * dstPtr,  rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    unsigned int padding = 10;
    unsigned int type = 1;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);


    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    //std::cout << "coming till here" << std::endl;
    handle.AddKernel("", "", "resize.cpp", "resize_crop_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        padding,
                                                                        type,
                                                                        plnpkdind);
    return RPP_SUCCESS;
}
/*************Rotate ******************/
RppStatus
rotate_hip(Rpp8u * srcPtr, RppiSize srcSize,
                Rpp8u * dstPtr, RppiSize dstSize, float angleDeg,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "rotate.cpp", "rotate_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        angleDeg,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "rotate.cpp", "rotate_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        angleDeg,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
    }
    return RPP_SUCCESS;
}
RppStatus
rotate_hip_batch (   Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "rotate.cpp", "rotate_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                channel,
                                                                handle.GetInitHandle()->mem.mgpu.inc,
                                                                handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                plnpkdind
                                                                );
    return RPP_SUCCESS;
}

/************* Warp affine ******************/
RppStatus
warp_affine_hip(Rpp8u * srcPtr, RppiSize srcSize,
                Rpp8u * dstPtr, RppiSize dstSize, float *affine,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{

    float affine_inv[6];
    float det; //for Deteminent
    det = (affine[0] * affine [4])  - (affine[1] * affine[3]);
    affine_inv[0] = affine[4]/ det;
    affine_inv[1] = (- 1 * affine[1])/ det;
    affine_inv[2] = -1 * affine[2];
    affine_inv[3] = (-1 * affine[3]) /det ;
    affine_inv[4] = affine[0]/det;
    affine_inv[5] = -1 * affine[5];

    float *affine_matrix;
    Rpp32u* affine_array;
    hipMalloc(&affine_array, sizeof(float)*6);
    hipMemcpy(affine_array,affine_inv,sizeof(float)*6,hipMemcpyHostToDevice);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        affine_array,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
        // CreateProgramFromBinary(handle,"warp_affine.cpp","warp_affine.cpp.bin","waro_affine_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
        // cl_kernel_initializer(  "rotate.cpp", "rotate_pln",
        //                         theProgram, theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        affine_array,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
        // CreateProgramFromBinary(handle,"warp_affine.cpp","warp_affine.cpp.bin","warp_affine_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
        // cl_kernel_initializer(  "rotate.cpp", "rotate_pkd",
        //                         theProgram, theKernel);
    }

    else
    {std::cerr << "Internal error: Unknown Channel format";}

    // err  = clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &affine_array);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &dstSize.height);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &dstSize.width);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);


    // size_t gDim3[3];
    // gDim3[0] = dstSize.width;
    // gDim3[1] = dstSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}

RppStatus
warp_affine_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle,Rpp32f *affine,
                        RppiChnFormat chnFormat, unsigned int channel)
    // Rpp8u* srcPtr, RppiSize *srcSize, RppiSize *src_maxSize,
    //                         Rpp8u* dstPtr, RppiSize *dstSize, RppiSize *dst_maxSize,
    //                         Rpp32f *affine, Rpp32u nBatchSize,
    //                         RppiChnFormat chnFormat, unsigned int channel,
    //                         rpp::Handle& handle)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    float affine_inv[6];
    float det; //for Deteminent
    short counter;
    size_t gDim3[3];
    Rpp32f* affine_array;
    hipMalloc(&affine_array, sizeof(float)*6);

    unsigned int maxsrcHeight, maxsrcWidth, maxdstHeight, maxdstWidth;
    maxsrcHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxsrcWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxdstHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[0];
    maxdstWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[0];
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxsrcHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxsrcHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxsrcWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxsrcWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        
        if(maxdstHeight < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
            maxdstHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        if(maxdstWidth < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
            maxdstWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
    }

    Rpp8u * srcPtr1;
    hipMalloc(&srcPtr1,sizeof(unsigned char) * maxsrcHeight * maxsrcWidth * channel);
    Rpp8u * dstPtr1;
    hipMalloc(&dstPtr1,sizeof(unsigned char) * maxdstHeight * maxdstWidth * channel);

    int ctr;
    size_t srcbatchIndex = 0, dstbatchIndex = 0;
    size_t index =0;

    for(int i =0; i<nBatchSize; i++)
    {
        hipMemcpy(srcPtr1, srcPtr+srcbatchIndex , sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);        det = (affine[index+0] * affine [index+4])  - (affine[index+1] * affine[index+3]);
        affine_inv[0] = affine[index+4]/ det;
        affine_inv[1] = (- 1 * affine[index+1])/ det;
        affine_inv[2] = -1 * affine[index+2];
        affine_inv[3] = (-1 * affine[index+3]) /det ;
        affine_inv[4] = affine[index+0]/det;
        affine_inv[5] = -1 * affine[index+5];

        hipMemcpy(affine_array,affine_inv,sizeof(float)*6,hipMemcpyHostToDevice);
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
        gDim3[1] = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        gDim3[2] = channel;
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        affine_array,
                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                        handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                        handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                        channel
                                                                        );
            // CreateProgramFromBinary(handle.GetStream(),"warp_affine.cpp","warp_affine.cpp.bin","warp_affine_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "warp_affine.cpp", "warp_affine_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        affine_array,
                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                        handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                        handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                        channel
                                                                        );
            // CreateProgramFromBinary(handle.GetStream(),"warp_affine.cpp","warp_affine.cpp.bin","warp_affine_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }

        else
        {std::cerr << "Internal error: Unknown Channel format";}

        // int ctr =0;
        // err  = clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr1);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr1);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &affine_array);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.height[i]);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.width[i]);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
        hipMemcpy(dstPtr+dstbatchIndex, dstPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);        srcbatchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        dstbatchIndex += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);
        index = index + 6;
    }
    return RPP_SUCCESS;
    /* CLrelease should come here */
}
/************* Warp Perspective ******************/
RppStatus
warp_perspective_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr,
                    RppiSize dstSize, float *perspective, RppiChnFormat chnFormat,
                    unsigned int channel, rpp::Handle& handle)
{

    float perspective_inv[9];
    float det; //for Deteminent
    det = (perspective[0] * ((perspective[4] * perspective[8]) - (perspective[5] * perspective[7]))) - (perspective[1] * ((perspective[3] * perspective[8]) - (perspective[5] * perspective[6]))) + (perspective[2] * ((perspective[3] * perspective[7]) - (perspective[4] * perspective[6])));
    perspective_inv[0] = (1 * ((perspective[4] * perspective[8]) - (perspective[5] * perspective[7]))) / det;
    perspective_inv[1] = (-1 * ((perspective[1] * perspective[8]) - (perspective[7] * perspective[2]))) / det;
    perspective_inv[2] = (1 * ((perspective[1] * perspective[5]) - (perspective[4] * perspective[2]))) / det;
    perspective_inv[3] = (-1 * ((perspective[3] * perspective[8]) - (perspective[6] * perspective[5]))) / det;
    perspective_inv[4] = (1 * ((perspective[0] * perspective[8]) - (perspective[6] * perspective[2]))) / det;
    perspective_inv[5] = (-1 * ((perspective[0] * perspective[5]) - (perspective[3] * perspective[2]))) / det;
    perspective_inv[6] = (1 * ((perspective[3] * perspective[7]) - (perspective[6] * perspective[4]))) / det;
    perspective_inv[7] = (-1 * ((perspective[0] * perspective[7]) - (perspective[6] * perspective[1]))) / det;
    perspective_inv[8] = (1 * ((perspective[0] * perspective[4]) - (perspective[3] * perspective[1]))) / det;

    float *perspective_matrix;
    Rpp32f* perspective_array;
    hipMalloc(&perspective_array, sizeof(float) * 9);
    hipMemcpy(perspective_array, perspective_inv,sizeof(float) * 9, hipMemcpyHostToDevice);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        perspective_array,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
        // CreateProgramFromBinary(handle.GetStream(),"warp_perspective.cpp","warp_perspective.cpp.bin","warp_perspective_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        perspective_array,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel
                                                                        );
        // CreateProgramFromBinary(handle.GetStream(),"warp_perspective.cpp","warp_perspective.cpp.bin","warp_perspective_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    // int ctr =0;
    // err  = clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &perspective_array);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &dstSize.height);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &dstSize.width);
    // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);


    // size_t gDim3[3];
    // gDim3[0] = dstSize.width;
    // gDim3[1] = dstSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}

RppStatus
warp_perspective_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr,  rpp::Handle& handle,Rpp32f *perspective, 
                        RppiChnFormat chnFormat, unsigned int channel)
    // Rpp8u* srcPtr, RppiSize *srcSize, RppiSize *src_maxSize,
    //                         Rpp8u* dstPtr, RppiSize *dstSize, RppiSize *dst_maxSize,
    //                         Rpp32f *perspective, Rpp32u nBatchSize,
    //                         RppiChnFormat chnFormat, unsigned int channel,
    //                         rpp::Handle& handle)
{
    Rpp32u nBatchSize = handle.GetBatchSize(); 
    float perspective_inv[9];
    float det; //for Deteminent
    short counter;
    size_t gDim3[3];
    Rpp32f* perspective_array;
    hipMalloc(&perspective_array,sizeof(float)*9);

    unsigned int maxsrcHeight, maxsrcWidth, maxdstHeight, maxdstWidth;
    maxsrcHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxsrcWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    maxdstHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[0];
    maxdstWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[0];
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxsrcHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxsrcHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxsrcWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxsrcWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        
        if(maxdstHeight < handle.GetInitHandle()->mem.mgpu.cdstSize.height[i])
            maxdstHeight = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        if(maxdstWidth < handle.GetInitHandle()->mem.mgpu.cdstSize.width[i])
            maxdstWidth = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
    }

    Rpp8u * srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxsrcHeight * maxsrcWidth * channel);
    Rpp8u * dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned char) * maxdstHeight * maxdstWidth * channel);

    int ctr;
    size_t srcbatchIndex = 0, dstbatchIndex = 0;
    size_t index =0;

    for(int i =0; i<nBatchSize; i++)
    {
        hipMemcpy(srcPtr1, srcPtr+srcbatchIndex , sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * 
        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        det = (perspective[index] * ((perspective[index+4] * perspective[index+8]) - (perspective[index+5] * perspective[index+7]))) - (perspective[index+1] * ((perspective[index+3] * perspective[index+8]) - (perspective[index+5] * perspective[index+6]))) + (perspective[index+2] * ((perspective[index+3] * perspective[index+7]) - (perspective[index+4] * perspective[index+6])));
        perspective_inv[0] = (1 * ((perspective[index+4] * perspective[index+8]) - (perspective[index+5] * perspective[index+7]))) / det;
        perspective_inv[1] = (-1 * ((perspective[index+1] * perspective[index+8]) - (perspective[index+7] * perspective[index+2]))) / det;
        perspective_inv[2] = (1 * ((perspective[index+1] * perspective[index+5]) - (perspective[index+4] * perspective[index+2]))) / det;
        perspective_inv[3] = (-1 * ((perspective[index+3] * perspective[index+8]) - (perspective[index+6] * perspective[index+5]))) / det;
        perspective_inv[4] = (1 * ((perspective[index] * perspective[index+8]) - (perspective[index+6] * perspective[index+2]))) / det;
        perspective_inv[5] = (-1 * ((perspective[index] * perspective[index+5]) - (perspective[index+3] * perspective[index+2]))) / det;
        perspective_inv[6] = (1 * ((perspective[index+3] * perspective[index+7]) - (perspective[index+6] * perspective[index+4]))) / det;
        perspective_inv[7] = (-1 * ((perspective[index] * perspective[index+7]) - (perspective[index+6] * perspective[index+1]))) / det;
        perspective_inv[8] = (1 * ((perspective[index] * perspective[index+4]) - (perspective[index+3] * perspective[index+1]))) / det;

        hipMemcpy(perspective_array,perspective_inv,sizeof(float)*9,hipMemcpyHostToDevice);

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.cdstSize.width[i], handle.GetInitHandle()->mem.mgpu.cdstSize.height[i], channel};
            handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pln", vld, vgd, "")(srcPtr1,
                                                                        dstPtr1,
                                                                        perspective_array,
                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                        handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                        handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                        channel
                                                                        );
            // CreateProgramFromBinary(handle.GetStream(),"warp_perspective.cpp","warp_perspective.cpp.bin","warp_perspective_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {   
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{handle.GetInitHandle()->mem.mgpu.cdstSize.width[i], handle.GetInitHandle()->mem.mgpu.cdstSize.height[i], channel};
            handle.AddKernel("", "", "warp_perspective.cpp", "warp_perspective_pkd", vld, vgd, "")(srcPtr1,
                                                                        dstPtr1,
                                                                        perspective_array,
                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                        handle.GetInitHandle()->mem.mgpu.cdstSize.height[i],
                                                                        handle.GetInitHandle()->mem.mgpu.cdstSize.width[i],
                                                                        channel
                                                                        );
            // CreateProgramFromBinary(handle.GetStream(),"warp_perspective.cpp","warp_perspective.cpp.bin","warp_perspective_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }

        else
        {std::cerr << "Internal error: Unknown Channel format";}

        // int ctr =0;
        // err  = clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr1);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr1);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &perspective_array);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.height[i]);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.cdstSize.width[i]);
        // err |= clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);

        // gDim3[0] = handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
        // gDim3[1] = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i];
        // gDim3[2] = channel;
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
        hipMemcpy(dstPtr+dstbatchIndex, dstPtr1, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);        srcbatchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        dstbatchIndex += handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] * channel * sizeof(unsigned char);
        index = index + 9;
    }
    return RPP_SUCCESS;
    /* CLrelease should come here */
}
// /************* Scale ******************/
RppStatus
scale_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize,
 Rpp32f percentage, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    percentage /= 100;
    unsigned int dstheight = (Rpp32s) (percentage * (Rpp32f) srcSize.height);
    unsigned int dstwidth = (Rpp32s) (percentage * (Rpp32f) srcSize.width);    
    unsigned short counter=0;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "scale.cpp", "scale_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel,
                                                                        dstheight,
                                                                        dstwidth
                                                                        );
        // CreateProgramFromBinary(handle.GetStream(),"resize.cpp","resize.cpp.bin","resize_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {   
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{dstSize.width, dstSize.height, channel};
        handle.AddKernel("", "", "scale.cpp", "scale_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        dstSize.height,
                                                                        dstSize.width,
                                                                        channel,
                                                                        dstheight,
                                                                        dstwidth
                                                                        );
        // CreateProgramFromBinary(handle.GetStream(),"resize.cpp","resize.cpp.bin","resize_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstheight);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dstwidth);


    // size_t gDim3[3];
    // gDim3[0] = dstSize.width;
    // gDim3[1] = dstSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}

RppStatus
scale_hip_batch (Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
    // Rpp8u* srcPtr, RppiSize *srcSize, RppiSize *src_maxSize,
    //                         Rpp8u* dstPtr, RppiSize *dstSize, RppiSize *dst_maxSize,
    //                          Rpp32f *percentage, Rpp32u nBatchSize,
    //                         RppiChnFormat chnFormat, unsigned int channel,
    //                         rpp::Handle& handle)
{
     int plnpkdind;

    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    unsigned int padding = 0;
    unsigned int type = 0;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    //std::cout << "coming till here" << std::endl;
    handle.AddKernel("", "", "scale.cpp", "scale_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,   
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        plnpkdind);
    return RPP_SUCCESS;
}