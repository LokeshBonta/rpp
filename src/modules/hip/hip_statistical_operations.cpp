#include "hip_declarations.hpp"
RppStatus
thresholding_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp8u min,
 Rpp8u max, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "thresholding.cpp", "thresholding", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel,
                                                                            min,
                                                                            max);
    
    // unsigned short counter=0;
    // cl_kernel theKernel;
    // cl_program theProgram;
    // CreateProgramFromBinary(theQueue,"thresholding.cpp","thresholding.cpp.bin","thresholding",theProgram,theKernel);
    // clRetainKernel(theKernel);
    
    // //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(Rpp8u*), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned char), &min);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned char), &max);
    // //----

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    // cl_kernel_implementer (theQueue, gDim3, NULL/*Local*/, theProgram, theKernel);

    return RPP_SUCCESS;
}

RppStatus
thresholding_hip_batch (   Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "thresholding.cpp", "thresholding_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                handle.GetInitHandle()->mem.mgpu.ucharArr[0].ucharmem,
                                                                handle.GetInitHandle()->mem.mgpu.ucharArr[1].ucharmem,
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

/********************** thresholding ************************/

RppStatus
min_hip ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,
                 RppiSize srcSize, Rpp8u* dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "min.cpp", "min_hip", vld, vgd, "")(srcPtr1,
                                                                                            srcPtr2,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;

}


RppStatus
min_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "min.cpp", "min_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
/********************** max ************************/

RppStatus
max_hip ( Rpp8u* srcPtr1,Rpp8u* srcPtr2,
                 RppiSize srcSize, Rpp8u* dstPtr,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};
    handle.AddKernel("", "", "max.cpp", "max_hip", vld, vgd, "")(srcPtr1,
                                                            srcPtr2,
                                                            dstPtr,
                                                            srcSize.height,
                                                            srcSize.width,
                                                            channel);

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = srcSize.height;
    // gDim3[2] = channel;
    return RPP_SUCCESS;

}


RppStatus
max_hip_batch ( Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle,
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
    handle.AddKernel("", "", "max.cpp", "max_batch", vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
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
/********************** min_max_loc ************************/
RppStatus
min_max_loc_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp32u* minLoc,
                Rpp32u* maxLoc, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int i;

    const int LIST_SIZE = srcSize.height * srcSize.width * channel;
    int numGroups = std::ceil(LIST_SIZE / 256);
    
    unsigned char minElement = 255;
    unsigned int minLocation;
    unsigned char *partial_min;
    partial_min = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    unsigned int *partial_min_location;
    partial_min_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));
    Rpp8u* b_mem_obj;
    hipMalloc(&b_mem_obj,numGroups * sizeof(unsigned char));
    hipMemcpy(b_mem_obj, partial_min, numGroups * sizeof(unsigned char),hipMemcpyHostToDevice);
    Rpp32u* b_mem_obj1;
    hipMalloc(&b_mem_obj1,numGroups * sizeof(unsigned int));
    hipMemcpy(b_mem_obj1, partial_min_location, numGroups * sizeof(unsigned int),hipMemcpyHostToDevice);

    unsigned char maxElement = 0;
    unsigned int maxLocation;
    unsigned char *partial_max;
    partial_max = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    unsigned int *partial_max_location;
    partial_max_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));
    Rpp8u* c_mem_obj;
    hipMalloc(&c_mem_obj, numGroups * sizeof(unsigned char));
    hipMemcpy(c_mem_obj,partial_max,numGroups * sizeof(unsigned char),hipMemcpyHostToDevice);
    Rpp32u* c_mem_obj1;
    hipMalloc(&c_mem_obj1,numGroups * sizeof(unsigned int));
    hipMemcpy(c_mem_obj1, partial_max_location, numGroups * sizeof(unsigned int),hipMemcpyHostToDevice);
    size_t gDim3[3];
    gDim3[0] = LIST_SIZE;
    gDim3[1] = 1;
    gDim3[2] = 1;
    size_t local_item_size[3];
    local_item_size[0] = 256;
    local_item_size[1] = 1;
    local_item_size[2] = 1;
    std::vector<size_t> vld{local_item_size[0], local_item_size[1], local_item_size[2]};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
    handle.AddKernel("", "", "min_max_loc.cpp", "min", vld, vgd, "")(srcPtr,
                                                                    b_mem_obj,
                                                                    b_mem_obj1);
    // CreateProgramFromBinary(handle.GetStream(),"min_max_loc.cpp","min_max_loc.cpp.bin","min",theProgram,theKernel);
    // clRetainKernel(theKernel);   
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj1);
    
    hipMemcpy(partial_min, b_mem_obj, numGroups * sizeof(unsigned char), hipMemcpyDeviceToHost);
    hipMemcpy(partial_min_location, b_mem_obj1, numGroups * sizeof(unsigned char),hipMemcpyDeviceToHost );   
    
    for(i = 0; i < numGroups; i++)
    {
        if(minElement > partial_min[i])
        {
            minElement = partial_min[i];
            minLocation = partial_min_location[i];

        }
    }
    *min = minElement;
    *minLoc=minLocation;

    std::vector<size_t> vld1{local_item_size[0], local_item_size[1], local_item_size[2]};
    std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};
    handle.AddKernel("", "", "min_max_loc.cpp", "max", vld1, vgd1, "")(srcPtr,
                                                                    c_mem_obj,
                                                                    c_mem_obj1);
    // CreateProgramFromBinary(handle.GetStream(),"min_max_loc.cpp","min_max_loc.cpp.bin","max",theProgram,theKernel);
    // clRetainKernel(theKernel); 

    // counter = 0;
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj1);

    // cl_kernel_implementer (gDim3, local_item_size, theProgram, theKernel);

    hipMemcpy(partial_max, c_mem_obj, numGroups * sizeof(unsigned char), hipMemcpyDeviceToHost); 
    hipMemcpy(partial_max_location, b_mem_obj1, numGroups * sizeof(unsigned char), hipMemcpyDeviceToHost); 
    for(i = 0; i < numGroups; i++)
    {
        if(maxElement < partial_max[i])
        {
            maxElement = partial_max[i];
            maxLocation = partial_max_location[i];
        }
    }

    *max = maxElement;
    *maxLoc=maxLocation;

    hipFree(b_mem_obj); 
    free(partial_min);
    hipFree(c_mem_obj); 
    free(partial_max);
    hipFree(b_mem_obj1); 
    free(partial_min_location);
    hipFree(c_mem_obj1); 
    free(partial_max_location);
    return RPP_SUCCESS;
}

RppStatus
min_max_loc_hip_batch(Rpp8u* srcPtr, Rpp8u *min, Rpp8u *max,
                    Rpp32u *minLoc, Rpp32u *maxLoc, rpp::Handle& handle,
                    RppiChnFormat chnFormat, unsigned int channel)
// cl_mem srcPtr, RppiSize *srcSize, Rpp8u *min, Rpp8u *max,
//  Rpp32u *minLoc, Rpp32u *maxLoc, Rpp32u nBatchSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    unsigned char *partial_min;
    unsigned int *partial_min_location;
    unsigned char *partial_max;
    unsigned int *partial_max_location;

    int numGroups = 0;
    
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        int size = 0;
        size = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] *
                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel;
        int group = std::ceil(size / 256);
        if(numGroups < group)
            numGroups = group;
    }

    partial_min = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    partial_min_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));
    partial_max = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    partial_max_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));

    Rpp8u* b_mem_obj;
    hipMalloc(&b_mem_obj, numGroups * sizeof(unsigned char));
    Rpp32u* b_mem_obj1;
    hipMalloc(&b_mem_obj1, numGroups * sizeof(unsigned int));
    Rpp8u* c_mem_obj;
    hipMalloc(&c_mem_obj, numGroups * sizeof(unsigned char));
    Rpp32u* c_mem_obj1;
    hipMalloc(&c_mem_obj1, numGroups * sizeof(unsigned int));

    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char)* maxHeight * maxWidth * channel);
    Rpp32u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned int)* maxHeight * maxWidth * channel);

    size_t gDim3[3];

    size_t batchIndex = 0;

    for(int x = 0 ; x < nBatchSize ; x++)
    {                
        hipMemcpy(srcPtr1, srcPtr+batchIndex ,sizeof(unsigned char) *
                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] *
                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * channel,hipMemcpyDeviceToDevice);

        int i;

        int LIST_SIZE = handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * channel;
        numGroups = std::ceil(LIST_SIZE / 256);

        unsigned char minElement = 255;
        unsigned int minLocation;

        hipMemcpy(b_mem_obj, partial_min,numGroups * sizeof(unsigned char), hipMemcpyHostToDevice);
        hipMemcpy(b_mem_obj1, partial_min_location, numGroups * sizeof(unsigned int), hipMemcpyHostToDevice);

        unsigned char maxElement = 0;
        unsigned int maxLocation;

        hipMemcpy(c_mem_obj, partial_max, numGroups * sizeof(unsigned char),hipMemcpyHostToDevice);
        hipMemcpy(c_mem_obj1, partial_max_location, numGroups * sizeof(unsigned int),hipMemcpyHostToDevice);

        size_t gDim3[3];
        gDim3[0] = LIST_SIZE;
        gDim3[1] = 1;
        gDim3[2] = 1;
        size_t local_item_size[3];
        local_item_size[0] = 256;
        local_item_size[1] = 1;
        local_item_size[2] = 1;
        std::vector<size_t> vld{local_item_size[0], local_item_size[1], local_item_size[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "min_max_loc.cpp", "min", vld, vgd, "")(srcPtr1,
                                                                    b_mem_obj,
                                                                    b_mem_obj1);
        // CreateProgramFromBinary(handle.GetStream(),"min_max_loc.cpp","min_max_loc.cpp.bin","min",theProgram,theKernel);
        // clRetainKernel(theKernel);   
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj1);
        // cl_kernel_implementer (gDim3, local_item_size, theProgram, theKernel);
        
        hipMemcpy(partial_min,b_mem_obj, numGroups * sizeof(unsigned char), hipMemcpyDeviceToHost);
        
        hipMemcpy(partial_min_location,b_mem_obj1, numGroups * sizeof(unsigned char),hipMemcpyDeviceToHost );   
        
        for(i = 0; i < numGroups; i++)
        {
            if(minElement > partial_min[i])
            {
                minElement = partial_min[i];
                minLocation = partial_min_location[i];

            }
        }
        *min = minElement;
        *minLoc = minLocation;

        std::vector<size_t> vld1{local_item_size[0], local_item_size[1], local_item_size[2]};
        std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "min_max_loc.cpp", "max", vld1, vgd1, "")(srcPtr1,
                                                                        c_mem_obj,
                                                                        c_mem_obj1);
        // CreateProgramFromBinary(handle.GetStream(),"min_max_loc.cpp","min_max_loc.cpp.bin","max",theProgram,theKernel);
        // clRetainKernel(theKernel); 

        // counter = 0;
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj1);

        // cl_kernel_implementer (gDim3, local_item_size, theProgram, theKernel);
        
        hipMemcpy(partial_max,c_mem_obj, numGroups * sizeof(unsigned char), hipMemcpyDeviceToHost); 
        
        hipMemcpy(partial_max_location,b_mem_obj1, numGroups * sizeof(unsigned char), hipMemcpyDeviceToHost); 
        
        for(i = 0; i < numGroups; i++)
        {
            if(maxElement < partial_max[i])
            {
                maxElement = partial_max[i];
                maxLocation = partial_max_location[i];
            }
        }
        *max = maxElement;
        *maxLoc = maxLocation;

        min++;
        max++;
        minLoc++;
        maxLoc++;
        
        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * channel * sizeof(unsigned char);
    }
    hipFree(b_mem_obj); 
    free(partial_min);
    hipFree(c_mem_obj); 
    free(partial_max);
    hipFree(b_mem_obj1); 
    free(partial_min_location);
    hipFree(c_mem_obj1); 
    free(partial_max_location);

    return RPP_SUCCESS;
}

/********************** Integral ************************/
RppStatus
integral_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp32u* dstPtr, RppiChnFormat chnFormat,
             unsigned int channel, rpp::Handle& handle)
{

    unsigned short counter=0;
    
    Rpp32u* hInput;
    hipMalloc(&hInput, sizeof(unsigned int)* srcSize.height * srcSize.width * channel);
    
    /* FIRST COLUMN */
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, 1, channel};
        handle.AddKernel("", "", "integral.cpp", "integral_pkd_col", vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);
        // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_pkd_col",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, 1, channel};
        handle.AddKernel("", "", "integral.cpp", "integral_pln_col", vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);
        // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_pln_col",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    
    //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // //----

    // size_t gDim3[3];
    // gDim3[0] = srcSize.width;
    // gDim3[1] = 1;
    // gDim3[2] = channel;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    /* FIRST ROW */
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.height, 1, channel};
        handle.AddKernel("", "", "integral.cpp", "integral_pkd_row", vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);
        // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_pkd_row",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.height, 1, channel};
        handle.AddKernel("", "", "integral.cpp", "integral_pln_row", vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                srcSize.height,
                                                                                srcSize.width,
                                                                                channel);
        // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_pln_row",theProgram,theKernel);
        // clRetainKernel(theKernel);
    }
    
    // counter=0;
    // //---- Args Setter
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
    // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
    // //----

    // gDim3[0] = srcSize.height;
    // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

    Rpp32u temp = 1;

    for(int i = 0 ; i < srcSize.height - 1 ; i++)
    {

        if(i + 1 < ((srcSize.height - 1 <= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1))
            temp = i + 1;
        else if(i >= ((srcSize.height - 1 >= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1))
            temp = srcSize.height - i + srcSize.width - 3;
        else
            temp = (srcSize.height - 1 <= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1;
        // gDim3[0] = temp;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{temp, 1, channel};
            handle.AddKernel("", "", "integral.cpp", "integral_up_pkd", vld, vgd, "")(srcPtr,
                                                                                    dstPtr,
                                                                                    srcSize.height,
                                                                                    srcSize.width,
                                                                                    channel,
                                                                                    i,
                                                                                    temp);
            // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_up_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{temp, 1, channel};
            handle.AddKernel("", "", "integral.cpp", "integral_up_pln", vld, vgd, "")(srcPtr,
                                                                                    dstPtr,
                                                                                    srcSize.height,
                                                                                    srcSize.width,
                                                                                    channel,
                                                                                    i,
                                                                                    temp);
            // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_up_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        counter=0;

        //---- Args Setter
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, counter++, sizeof(int), &i);
        // //----
        // clSetKernelArg(theKernel, counter++, sizeof(int), &temp);
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    }
    for(int i = 0 ; i < srcSize.width - 2 ; i++)
    {

        if(i + 1 + srcSize.height - 1 < ((srcSize.height - 1 <= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1))
            temp = i + 1;
        else if(i + srcSize.height - 1 >= ((srcSize.height - 1 >= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1))
            temp = srcSize.height - (i + srcSize.height - 1) + srcSize.width - 3;
        else
            temp = (srcSize.height - 1 <= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1;
        

        // gDim3[0] = temp;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{temp, 1, channel};
            handle.AddKernel("", "", "integral.cpp", "integral_low_pkd", vld, vgd, "")(srcPtr,
                                                                                    dstPtr,
                                                                                    srcSize.height,
                                                                                    srcSize.width,
                                                                                    channel,
                                                                                    i,
                                                                                    temp);
            // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_low_pkd",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{temp, 1, channel};
            handle.AddKernel("", "", "integral.cpp", "integral_low_pln", vld, vgd, "")(srcPtr,
                                                                                    dstPtr,
                                                                                    srcSize.height,
                                                                                    srcSize.width,
                                                                                    channel,
                                                                                    i,
                                                                                    temp);
            // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_low_pln",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        counter=0;

        //---- Args Setter
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
        // clSetKernelArg(theKernel, counter++, sizeof(int), &i);
        // //----
        // clSetKernelArg(theKernel, counter++, sizeof(int), &temp);
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
    }
    return RPP_SUCCESS;
}

RppStatus
integral_hip_batch(Rpp8u* srcPtr, Rpp32u* dstPtr, rpp::Handle& handle,
                    RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nBatchSize = handle.GetBatchSize();

    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }

    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    Rpp32u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned int) * maxHeight * maxWidth * channel);

    int counter;
           
    size_t gDim3[3];

    size_t batchIndexSrc = 0;
    size_t batchIndexDst = 0;

    for(int i = 0 ; i < nBatchSize ; i++)
    {                
        hipMemcpy(srcPtr1, srcPtr+batchIndexSrc , sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);        /* FIRST COLUMN */
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = 1;
        gDim3[2] = channel;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cpp", "integral_pkd_col", vld, vgd, "")(srcPtr1,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_pkd_col",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cpp", "integral_pln_col", vld, vgd, "")(srcPtr1,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_pln_col",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        //---- Args Setter
        // counter = 0;
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr1);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
        // //----


        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

        /* FIRST ROW */
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cpp", "integral_pkd_row", vld, vgd, "")(srcPtr1,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_pkd_row",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cpp", "integral_pln_row", vld, vgd, "")(srcPtr1,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                channel);
            // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_pln_row",theProgram,theKernel);
            // clRetainKernel(theKernel);
        }
        
        // counter=0;
        // //---- Args Setter
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr1);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
        // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
        // //----

        // gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);

        Rpp32u temp = 1;

        for(int x = 0 ; x < handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] - 1 ; x++)
        {
            if(x + 1 < ((handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 <= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1))
                temp = x + 1;
            else if(x >= ((handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 >= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1))
                temp = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - x + handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 3;
            else
                temp = (handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 <= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1;
            gDim3[0] = temp;
            if(chnFormat == RPPI_CHN_PACKED)
            {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cpp", "integral_up_pkd", vld, vgd, "")(srcPtr1,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                channel,
                                                                                x,
                                                                                temp);
                // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_up_pkd",theProgram,theKernel);
                // clRetainKernel(theKernel);
            }
            else
            {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cpp", "integral_up_pln", vld, vgd, "")(srcPtr1,
                                                                                dstPtr1,
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                channel,
                                                                                x,
                                                                                temp);
                // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_up_pln",theProgram,theKernel);
                // clRetainKernel(theKernel);
            }
            
            // counter=0;

            // //---- Args Setter
            // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
            // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr1);
            // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
            // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
            // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
            // clSetKernelArg(theKernel, counter++, sizeof(int), &x);
            //----
            // clSetKernelArg(theKernel, counter++, sizeof(int), &temp);
            // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
        }
        for(int x = 0 ; x < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 2 ; x++)
        {
            if(x + 1 + handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 < ((handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 <= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1))
                temp = x + 1;
            else if(x + handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 >= ((handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 >= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1))
                temp = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - (x + handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1) + handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 3;
            else
                temp = (handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 <= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1;
            gDim3[0] = temp;
            if(chnFormat == RPPI_CHN_PACKED)
            {
                std::vector<size_t> vld{32, 32, 1};
                std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
                handle.AddKernel("", "", "integral.cpp", "integral_low_pkd", vld, vgd, "")(srcPtr1,
                                                                                        dstPtr1,
                                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                        channel,
                                                                                        x,
                                                                                        temp);
                // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_low_pkd",theProgram,theKernel);
                // clRetainKernel(theKernel);
            }
            else
            {
                std::vector<size_t> vld{32, 32, 1};
                std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
                handle.AddKernel("", "", "integral.cpp", "integral_low_pln", vld, vgd, "")(srcPtr1,
                                                                                        dstPtr1,
                                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                        handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                        channel,
                                                                                        x,
                                                                                        temp);
                // CreateProgramFromBinary(handle.GetStream(),"integral.cpp","integral.cpp.bin","integral_low_pln",theProgram,theKernel);
                // clRetainKernel(theKernel);
            }
            
            counter=0;

            //---- Args Setter
            // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr1);
            // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr1);
            // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.height[i]);
            // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &handle.GetInitHandle()->mem.mgpu.csrcSize.width[i]);
            // clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
            // clSetKernelArg(theKernel, counter++, sizeof(int), &x);
            // //----


            // clSetKernelArg(theKernel, counter++, sizeof(int), &temp);
            // cl_kernel_implementer (gDim3, NULL/*Local*/, theProgram, theKernel);
        }
        hipMemcpy(dstPtr+batchIndexDst, dstPtr1, sizeof(unsigned int) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, hipMemcpyDeviceToDevice);
        batchIndexSrc += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        batchIndexDst += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned int);
    }
    return RPP_SUCCESS;
}
// /********************** Mean std dev ************************/

RppStatus
mean_stddev_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev,
                RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int i;
    
    const int LIST_SIZE = srcSize.height * srcSize.width * channel;
    int numGroups = std::ceil(LIST_SIZE / 256);
    float sum = 0;
    long *partial_sum;
    partial_sum = (long *) calloc (numGroups, sizeof(long));
    long* b_mem_obj;
    hipMalloc(&b_mem_obj, numGroups * sizeof(long));
    hipMemcpy(b_mem_obj, partial_sum, numGroups * sizeof(long), hipMemcpyHostToDevice);

    float mean_sum = 0;
    float *partial_mean_sum;
    partial_mean_sum = (float *) calloc (numGroups, sizeof(float));
    Rpp32f* c_mem_obj;
    hipMalloc(&c_mem_obj, numGroups * sizeof(float));
    hipMemcpy(c_mem_obj, partial_mean_sum, numGroups * sizeof(float),hipMemcpyHostToDevice);

    size_t gDim3[3];
    gDim3[0] = LIST_SIZE;
    gDim3[1] = 1;
    gDim3[2] = 1;
    size_t local_item_size[3];
    local_item_size[0] = 256;
    local_item_size[1] = 1;
    local_item_size[2] = 1;
    std::vector<size_t> vld{local_item_size[0], local_item_size[1], local_item_size[2]};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
    handle.AddKernel("", "", "mean_stddev.cpp", "sum", vld, vgd, "")(srcPtr,
                                                                    b_mem_obj);
    // CreateProgramFromBinary(handle.GetStream(),"mean_stddev.cpp","mean_stddev.cpp.bin","sum",theProgram,theKernel);
    // clRetainKernel(theKernel);   
    
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj);

    // cl_kernel_implementer (gDim3, local_item_size, theProgram, theKernel);
    
    hipMemcpy(partial_sum,b_mem_obj, numGroups * sizeof(long), hipMemcpyDeviceToHost);   
    
    for(i = 0; i < numGroups; i++)
    {
        sum += (float)partial_sum[i];
    }

    *mean = (sum) / LIST_SIZE ;

    float meanCopy = *mean;
    handle.AddKernel("", "", "mean_stddev.cpp", "mean_stddev", vld, vgd, "")(srcPtr,
                                                                    c_mem_obj,
                                                                    meanCopy);
    // CreateProgramFromBinary(handle.GetStream(),"mean_stddev.cpp","mean_stddev.cpp.bin","mean_stddev",theProgram,theKernel);
    // clRetainKernel(theKernel); 

    // counter = 0;
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
    // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj);
    // clSetKernelArg(theKernel, counter++, sizeof(float), &meanCopy);
    // cl_kernel_implementer (gDim3, local_item_size, theProgram, theKernel);
    hipMemcpy(partial_mean_sum,c_mem_obj, numGroups * sizeof(float), hipMemcpyDeviceToHost);  
    for(i = 0; i < numGroups; i++)
    {
        mean_sum += partial_mean_sum[i];
    }
    
    mean_sum = mean_sum / LIST_SIZE ;
    *stddev = mean_sum;

    hipFree(b_mem_obj); 
    free(partial_sum);
    hipFree(c_mem_obj); 
    free(partial_mean_sum);
    return RPP_SUCCESS;
}

RppStatus
mean_stddev_hip_batch(Rpp8u* srcPtr, Rpp32f *mean, Rpp32f *stddev, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
    // cl_mem srcPtr, RppiSize *srcSize, Rpp32f *mean, Rpp32f *stddev, Rpp32u nBatchSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    long *partial_sum;
    float *partial_mean_sum;

    int numGroups = 0;
    
    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    
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

    partial_sum = (long *) calloc (numGroups, sizeof(long));
    partial_mean_sum = (float *) calloc (numGroups, sizeof(float));

    long* b_mem_obj;
    hipMalloc(&b_mem_obj, numGroups * sizeof(long));
    Rpp32f* c_mem_obj;
    hipMalloc(&c_mem_obj, numGroups * sizeof(float));

    Rpp8u* srcPtr1;
    hipMalloc(&srcPtr1, sizeof(unsigned char) * maxHeight * maxWidth * channel);
    Rpp32u* dstPtr1;
    hipMalloc(&dstPtr1, sizeof(unsigned int) * maxHeight * maxWidth * channel);
    size_t gDim3[3];

    size_t batchIndex = 0;

    for(int x = 0 ; x < nBatchSize ; x++)
    {                
        hipMemcpy(srcPtr1, srcPtr+batchIndex , sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * channel, hipMemcpyDeviceToDevice);
        int i;

        int LIST_SIZE = handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * channel;
        numGroups = std::ceil(LIST_SIZE / 256);
        
        float sum = 0;
        hipMemcpy(b_mem_obj,  partial_sum, numGroups * sizeof(long),hipMemcpyHostToDevice);

        float mean_sum = 0;
        hipMemcpy(c_mem_obj,  partial_mean_sum,numGroups * sizeof(float),hipMemcpyHostToDevice);

        size_t gDim3[3];
        gDim3[0] = LIST_SIZE;
        gDim3[1] = 1;
        gDim3[2] = 1;
        size_t local_item_size[3];
        local_item_size[0] = 256;
        local_item_size[1] = 1;
        local_item_size[2] = 1;
        std::vector<size_t> vld{local_item_size[0], local_item_size[1], local_item_size[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        handle.AddKernel("", "", "mean_stddev.cpp", "sum", vld, vgd, "")(srcPtr1,
                                                                        b_mem_obj);
        // CreateProgramFromBinary(handle.GetStream(),"mean_stddev.cpp","mean_stddev.cpp.bin","sum",theProgram,theKernel);
        // clRetainKernel(theKernel);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &b_mem_obj);
        // cl_kernel_implementer (gDim3, local_item_size, theProgram, theKernel);
        
        hipMemcpy( partial_sum,b_mem_obj, numGroups * sizeof(long),hipMemcpyDeviceToHost);   
        
        for(i = 0; i < numGroups; i++)
        {
            sum += (float)partial_sum[i];
        }

        *mean = (sum) / LIST_SIZE ;
        float meanCopy = *mean;
        handle.AddKernel("", "", "mean_stddev.cpp", "sum", vld, vgd, "")(srcPtr1,
                                                                        c_mem_obj,
                                                                        meanCopy);

        // CreateProgramFromBinary(handle.GetStream(),"mean_stddev.cpp","mean_stddev.cpp.bin","mean_stddev",theProgram,theKernel);
        // clRetainKernel(theKernel); 
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
        // clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &c_mem_obj);
        // clSetKernelArg(theKernel, counter++, sizeof(float), &meanCopy);
        
        // cl_kernel_implementer (gDim3, local_item_size, theProgram, theKernel);
        hipMemcpy(partial_mean_sum,c_mem_obj, numGroups * sizeof(float),hipMemcpyDeviceToHost );  
        for(i = 0; i < numGroups; i++)
        {
            mean_sum += partial_mean_sum[i];
        }
        
        mean_sum = mean_sum / LIST_SIZE ;
        *stddev = mean_sum;

        stddev++;
        mean++;

        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * channel * sizeof(unsigned char);
    }

    hipFree(b_mem_obj); 
    free(partial_sum);
    hipFree(c_mem_obj); 
    free(partial_mean_sum);

    return RPP_SUCCESS;
}