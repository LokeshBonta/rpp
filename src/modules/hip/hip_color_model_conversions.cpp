#include "hip_declarations.hpp"

/****************  color_temperature modification *******************/

RppStatus
color_temperature_hip( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32s adjustmentValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "color_temperature.cpp", "temperature_planar", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            adjustmentValue
                                                                                            );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "color_temperature.cpp", "temperature_packed", vld, vgd, "")(srcPtr,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel,
                                                                                            adjustmentValue
                                                                                            );
    }
    //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, counter++, sizeof(float), &adjustmentValue);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = 1;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    
    return RPP_SUCCESS;
}

RppStatus
color_temperature_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "color_temperature.cpp", "color_temperature_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
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


/****************  Vignette modification *******************/

RppStatus
vignette_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, float stdDev, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int ctr=0;
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PLANAR)
    {    
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
        handle.AddKernel("", "", "vignette.cpp", "vignette_pln", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel,
                                                                            stdDev
                                                                            );
    } 
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};
        handle.AddKernel("", "", "vignette.cpp", "vignette_pkd", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel,
                                                                            stdDev
                                                                            );
    }
    
    // //---- Args Setter
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, ctr++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, ctr++, sizeof(float), &stdDev);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;
}


RppStatus
vignette_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "vignette.cpp", "vignette_batch", vld, vgd, "")(srcPtr, dstPtr,
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


/****************  channel_extract modification *******************/

RppStatus
channel_extract_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u extractChannelNumber, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "channel_extract.cpp", "channel_extract_pln", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel,
                                                                            extractChannelNumber
                                                                            );
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "channel_extract.cpp", "channel_extract_pkd", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel,
                                                                            extractChannelNumber
                                                                            );
    }
    //---- Args Setter
    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &extractChannelNumber);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = 1;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);
    return RPP_SUCCESS;  
}

RppStatus
channel_extract_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "channel_extract.cpp", "channel_extract_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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
/****************  channel_combine modification *******************/

RppStatus
channel_combine_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* srcPtr3, RppiSize srcSize,
 Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "channel_combine.cpp", "channel_combine_pln", vld, vgd, "")(srcPtr1,
                                                                                            srcPtr2,
                                                                                            srcPtr3,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel
                                                                                            );
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "channel_combine.cpp", "channel_combine_pkd", vld, vgd, "")(srcPtr1,
                                                                                            srcPtr2,
                                                                                            srcPtr3,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel
                                                                                            );
    }
    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = 1;
    return RPP_SUCCESS;      
}


RppStatus
channel_combine_hip_batch ( Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* srcPtr3, Rpp8u* dstPtr,rpp::Handle& handle,
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
    handle.AddKernel("", "", "channel_combine.cpp", "channel_combine_batch", vld, vgd, "")(srcPtr1, srcPtr2, srcPtr3, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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
/****************  Hue and Saturation modification *******************/
RppStatus
hueRGB_hip ( Rpp8u* srcPtr,RppiSize srcSize,
                 Rpp8u* dstPtr, float hue_factor,
                RppiChnFormat chnFormat, unsigned int channel,
                rpp::Handle& handle)
{
                   float sat = 0.0;
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};

    if (chnFormat == RPPI_CHN_PLANAR)
    {
       handle.AddKernel("", "", "hue.cpp", "huergb_pln", vld, vgd, "")(srcPtr, dstPtr, hue_factor, sat, srcSize.height, srcSize.width);
    }
    else
    {
        handle.AddKernel("", "", "hue.cpp", "huergb_pkd", vld, vgd, "")(srcPtr, dstPtr, hue_factor, sat, srcSize.height, srcSize.width);
    }
   
    return RPP_SUCCESS;
}

RppStatus
saturationRGB_hip ( Rpp8u* srcPtr,RppiSize srcSize,
                 Rpp8u* dstPtr, float sat,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle){
    float hue_factor = 0.0;
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};

    if (chnFormat == RPPI_CHN_PLANAR)
    {
       handle.AddKernel("", "", "hue.cpp", "huergb_pln", vld, vgd, "")(srcPtr, dstPtr, hue_factor, sat, srcSize.height, srcSize.width);
    }
    else
    {
        handle.AddKernel("", "", "hue.cpp", "huergb_pkd", vld, vgd, "")(srcPtr, dstPtr, hue_factor, sat, srcSize.height, srcSize.width);
    }
   
    return RPP_SUCCESS;
}

RppStatus
hueRGB_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "hue.cpp", "hue_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind
                                                                                        );
    return RPP_SUCCESS;
}

RppStatus
saturationRGB_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "hue.cpp", "saturation_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind
                                                                                        );
    return RPP_SUCCESS;
}
/****************  Look_up_table modification *******************/
RppStatus
look_up_table_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,Rpp8u* lutPtr,
 RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp8u* clLutPtr;
    hipMalloc(&clLutPtr,sizeof(Rpp8u)*256*channel);
    hipMemcpy(clLutPtr, lutPtr, sizeof(Rpp8u)*256*channel,hipMemcpyHostToDevice);

    unsigned short counter=0;
    cl_int err;
    cl_kernel theKernel;
    cl_program theProgram;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "look_up_table.cpp", "look_up_table_pln", vld, vgd, "")(srcPtr,
                                                                    dstPtr,
                                                                    clLutPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel
                                                                    );
        // CreateProgramFromBinary(theQueue,"look_up_table.cpp","look_up_table.cpp.bin","look_up_table_pln",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};
        handle.AddKernel("", "", "look_up_table.cpp", "look_up_table_pkd", vld, vgd, "")(srcPtr,
                                                                    dstPtr,
                                                                    clLutPtr,
                                                                    srcSize.height,
                                                                    srcSize.width,
                                                                    channel
                                                                    );
        // CreateProgramFromBinary(theQueue,"look_up_table.cpp","look_up_table.cpp.bin","look_up_table_pkd",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    //---- Args Setter
    // err  = clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &clLutPtr);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);   
    return RPP_SUCCESS;      
}
RppStatus
look_up_table_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp8u* lutPtr,rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)

{
    Rpp8u* clLutPtr;
    hipMalloc(&clLutPtr,sizeof(Rpp8u)*256*channel*handle.GetBatchSize());
    hipMemcpy(clLutPtr,  lutPtr,sizeof(Rpp8u)*256*channel,hipMemcpyHostToDevice);
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "look_up_table.cpp", "look_up_table_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        clLutPtr,
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
/****************  Tensor Look_up_table modification *******************/
RppStatus
tensor_look_up_table_hip(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, 
                        Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp8u* lutPtr, rpp::Handle& handle)
{
    Rpp8u* clLutPtr;
    hipMalloc(&clLutPtr,sizeof(Rpp8u)*256);
    hipMemcpy(clLutPtr, lutPtr, sizeof(Rpp8u)*256,hipMemcpyHostToDevice);
    size_t gDim3[3];
    if(tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if(tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for(int i = 2 ; i < tensorDimension ; i++)
        {    
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    
    unsigned int dim1,dim2,dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
    handle.AddKernel("", "", "tensor.cpp", "tensor_look_up_table", vld, vgd, "")(tensorDimension,
                                                                                srcPtr,
                                                                                dstPtr,
                                                                                dim1,
                                                                                dim2,
                                                                                dim3,
                                                                                clLutPtr);
    //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &tensorDimension);
    // clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim1);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim2);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &dim3);
    // err |= clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &clLutPtr);
    // CreateProgramFromBinary(theQueue,"look_up_table.cpp","look_up_table.cpp.bin","tensor_look_up_table",theProgram,theKernel);
    // clRetainKernel(theKernel);
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);   
    return RPP_SUCCESS;      
}