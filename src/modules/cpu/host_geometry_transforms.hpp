#include <cpu/rpp_cpu_common.hpp>

/**************** flip ***************/

template <typename T>
RppStatus flip_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                          Rpp32u *batch_flipAxis, RppiROI *roiPoints,
                          Rpp32u nbatchSize,
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32u flipAxis = batch_flipAxis[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            if (flipAxis == RPPI_HORIZONTAL_AXIS)
            {
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel, *dstPtrChannel, *srcPtrChannelROI;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * imageDimMax) + (y2 * batch_srcSizeMax[batchCount].width) + x1;

                    // #pragma omp parallel for
                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        
                        if (!((y1 <= i) && (i <= y2)))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                            srcPtrTemp += batch_srcSizeMax[batchCount].width;
                            dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        }
                        else
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                            srcPtrTemp += x1;
                            dstPtrTemp += x1;

                            memcpy(dstPtrTemp, srcPtrChannelROI, roiPoints[batchCount].roiWidth * sizeof(T));
                            srcPtrTemp += roiPoints[batchCount].roiWidth;
                            dstPtrTemp += roiPoints[batchCount].roiWidth;
                            srcPtrChannelROI -= batch_srcSizeMax[batchCount].width;

                            memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                            srcPtrTemp += remainingElementsAfterROI;
                            dstPtrTemp += remainingElementsAfterROI;
                        }
                    }
                }
            }
            else if (flipAxis == RPPI_VERTICAL_AXIS)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + roiPoints[batchCount].roiWidth;
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel, *dstPtrChannel, *srcPtrChannelROI;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * imageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;

                    // #pragma omp parallel for
                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        
                        if (!((y1 <= i) && (i <= y2)))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                            srcPtrTemp += batch_srcSizeMax[batchCount].width;
                            dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        }
                        else
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                            srcPtrTemp += x1;
                            dstPtrTemp += x1;

                            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                            Rpp32u alignedLength = bufferLength & ~15;

                            __m128i px0;
                            __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                            int vectorLoopCount = 0;
                            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                            {
                                srcPtrChannelROI -= 15;
                                px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                                px0 = _mm_shuffle_epi8(px0, vMask);
                                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                                srcPtrChannelROI -= 1;
                                dstPtrTemp += 16;
                            }
                            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                            {
                                *dstPtrTemp++ = (T) *srcPtrChannelROI--;
                            }
                            
                            srcPtrTemp += roiPoints[batchCount].roiWidth;
                            srcPtrChannelROI += srcROIIncrement;

                            memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                            srcPtrTemp += remainingElementsAfterROI;
                            dstPtrTemp += remainingElementsAfterROI;
                        }
                    }
                }
            }
            else if (flipAxis == RPPI_BOTH_AXIS)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width - roiPoints[batchCount].roiWidth;
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel, *dstPtrChannel, *srcPtrChannelROI;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * imageDimMax) + (y2 * batch_srcSizeMax[batchCount].width) + x2;

                    // #pragma omp parallel for
                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        
                        if (!((y1 <= i) && (i <= y2)))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                            srcPtrTemp += batch_srcSizeMax[batchCount].width;
                            dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        }
                        else
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                            srcPtrTemp += x1;
                            dstPtrTemp += x1;

                            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                            Rpp32u alignedLength = bufferLength & ~15;

                            __m128i px0;
                            __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                            int vectorLoopCount = 0;
                            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                            {
                                srcPtrChannelROI -= 15;
                                px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                                px0 = _mm_shuffle_epi8(px0, vMask);
                                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                                srcPtrChannelROI -= 1;
                                dstPtrTemp += 16;
                            }
                            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                            {
                                *dstPtrTemp++ = (T) *srcPtrChannelROI--;
                            }
                            
                            srcPtrTemp += roiPoints[batchCount].roiWidth;
                            srcPtrChannelROI -= srcROIIncrement;

                            memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                            srcPtrTemp += remainingElementsAfterROI;
                            dstPtrTemp += remainingElementsAfterROI;
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            
            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32u flipAxis = batch_flipAxis[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * roiPoints[batchCount].roiWidth;
            
            if (flipAxis == RPPI_HORIZONTAL_AXIS)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y2 * elementsInRowMax) + (x1 * channel);

                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                        dstPtrTemp += elementsInRowMax;
                        srcPtrTemp += elementsInRowMax;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                        srcPtrTemp += elementsBeforeROI;
                        dstPtrTemp += elementsBeforeROI;

                        memcpy(dstPtrTemp, srcPtrROI, elementsInRowROI * sizeof(T));
                        srcPtrTemp += elementsInRowROI;
                        dstPtrTemp += elementsInRowROI;
                        srcPtrROI -= elementsInRowMax;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
            else if (flipAxis == RPPI_VERTICAL_AXIS)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * elementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = elementsInRowMax + elementsInRowROI;
                
                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                        dstPtrTemp += elementsInRowMax;
                        srcPtrTemp += elementsInRowMax;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                        srcPtrTemp += elementsBeforeROI;
                        dstPtrTemp += elementsBeforeROI;

                        Rpp32u bufferLength = elementsInRowROI;
                        Rpp32u alignedLength = bufferLength & ~15;

                        __m128i px0;
                        __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                        {
                            srcPtrROI -= 13;
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrROI -= 2;
                            dstPtrTemp += 15;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                        {
                            memcpy(dstPtrTemp, srcPtrROI, channel * sizeof(T));
                            dstPtrTemp += channel;
                            srcPtrROI -= channel;
                        }

                        srcPtrTemp += elementsInRowROI;
                        srcPtrROI += srcROIIncrement;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
            else if (flipAxis == RPPI_BOTH_AXIS)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y2 * elementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = elementsInRowMax - elementsInRowROI;
                
                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                        dstPtrTemp += elementsInRowMax;
                        srcPtrTemp += elementsInRowMax;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                        srcPtrTemp += elementsBeforeROI;
                        dstPtrTemp += elementsBeforeROI;

                        Rpp32u bufferLength = elementsInRowROI;
                        Rpp32u alignedLength = bufferLength & ~15;

                        __m128i px0;
                        __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                        {
                            srcPtrROI -= 13;
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrROI -= 2;
                            dstPtrTemp += 15;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                        {
                            memcpy(dstPtrTemp, srcPtrROI, channel * sizeof(T));
                            dstPtrTemp += channel;
                            srcPtrROI -= channel;
                        }

                        srcPtrTemp += elementsInRowROI;
                        srcPtrROI -= srcROIIncrement;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus flip_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                    Rpp32u flipAxis,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + ((c + 1) * srcSize.height * srcSize.width) - srcSize.width;
                for (int i = 0; i < srcSize.height; i++)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    dstPtrTemp += srcSize.width;
                    srcPtrTemp -= srcSize.width;
                }
            }
        }
        else if (flipAxis == RPPI_VERTICAL_AXIS)
        {
            T *srcPtrTemp2;
            srcPtrTemp2 = srcPtr;

            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width) + srcSize.width - 1;
                for (int i = 0; i < srcSize.height; i++)
                {
                    srcPtrTemp2 = srcPtrTemp;

                    Rpp32u bufferLength = srcSize.width;
                    Rpp32u alignedLength = bufferLength & ~15;

                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        srcPtrTemp2 -= 15;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp2);
                        px0 = _mm_shuffle_epi8(px0, vMask);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrTemp2 -= 1;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (T) *srcPtrTemp2--;
                    }
                    srcPtrTemp = srcPtrTemp + srcSize.width;
                }
            }
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {
            T *srcPtrTemp2;
            srcPtrTemp2 = srcPtr;

            for (int c = 0; c < channel; c++)
            {
                srcPtrTemp = srcPtr + ((c+1) * srcSize.height * srcSize.width) - 1;
                for (int i = 0; i < srcSize.height; i++)
                {
                    srcPtrTemp2 = srcPtrTemp;

                    Rpp32u bufferLength = srcSize.width;
                    Rpp32u alignedLength = bufferLength & ~15;

                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        srcPtrTemp2 -= 15;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp2);
                        px0 = _mm_shuffle_epi8(px0, vMask);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrTemp2 -= 1;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (T) *srcPtrTemp2--;
                    }
                    srcPtrTemp = srcPtrTemp - srcSize.width;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
            srcPtrTemp = srcPtr + (channel * ((srcSize.height * srcSize.width) - srcSize.width));
            for (int i = 0; i < srcSize.height; i++)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                dstPtrTemp += elementsInRow;
                srcPtrTemp -= elementsInRow;
            }
        }
        else if (flipAxis == RPPI_VERTICAL_AXIS)
        {
            srcPtrTemp = srcPtr + (channel * (srcSize.width - 1));
            for (int i = 0; i < srcSize.height; i++)
            {
                Rpp32u bufferLength = channel * srcSize.width;
                Rpp32u alignedLength = bufferLength & ~15;

                __m128i px0;
                __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                {
                    srcPtrTemp -= 13;
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                    px0 = _mm_shuffle_epi8(px0, vMask);
                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrTemp -= 2;
                    dstPtrTemp += 15;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));
                    dstPtrTemp += channel;
                    srcPtrTemp -= channel;
                }

                srcPtrTemp = srcPtrTemp + (channel * (2 * srcSize.width));
            }
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {
            srcPtrTemp = srcPtr + (channel * ((srcSize.height * srcSize.width) - 1));
            for (int i = 0; i < srcSize.height; i++)
            {
                Rpp32u bufferLength = channel * srcSize.width;
                Rpp32u alignedLength = bufferLength & ~15;

                __m128i px0;
                __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                {
                    srcPtrTemp -= 13;
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                    px0 = _mm_shuffle_epi8(px0, vMask);
                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrTemp -= 2;
                    dstPtrTemp += 15;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));
                    dstPtrTemp += channel;
                    srcPtrTemp -= channel;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

/**************** fisheye ***************/

template <typename T>
RppStatus fisheye_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiROI *roiPoints,
                             Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }
            
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32f newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].height))) - 1.0;
                        Rpp32f newIsquared = newI * newI;
                        
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32f newJ = (((Rpp32f) (j * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].width))) - 1.0;
                                Rpp32f newJsquared = newJ * newJ;
                                Rpp32f euclideanDistance = sqrt(newIsquared + newJsquared);

                                if (euclideanDistance >= 0 && euclideanDistance <= 1)
                                {
                                    Rpp32f newEuclideanDistance = sqrt(1.0 - (euclideanDistance * euclideanDistance));
                                    newEuclideanDistance = (euclideanDistance + (1.0 - newEuclideanDistance)) / 2.0;
                                    if (newEuclideanDistance <= 1.0)
                                    {
                                        Rpp32f theta = atan2(newI, newJ);

                                        Rpp32f newIsrc = newEuclideanDistance * sin(theta);
                                        Rpp32f newJsrc = newEuclideanDistance * cos(theta);

                                        int iSrc = (int) (((newIsrc + 1.0) * ((Rpp32f) batch_srcSize[batchCount].height)) / 2.0);
                                        int jSrc = (int) (((newJsrc + 1.0) * ((Rpp32f) batch_srcSize[batchCount].width)) / 2.0);

                                        int srcPosition = (int)((iSrc * batch_srcSize[batchCount].width) + jSrc);

                                        if ((srcPosition >= 0) && (srcPosition < imageDim))
                                        {
                                            int srcPositionTrue = (int)((iSrc * batch_srcSizeMax[batchCount].width) + jSrc);
                                            *dstPtrTemp = *(srcPtrTemp + srcPositionTrue);
                                        }
                                        else
                                        {
                                            *dstPtrTemp = (T) 0;
                                        }
                                    }
                                }
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            
            // #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].height))) - 1.0;
                Rpp32f newIsquared = newI * newI;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, elementsInRow * sizeof(T));
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32f newJ = (((Rpp32f) (j * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].width))) - 1.0;
                            Rpp32f newJsquared = newJ * newJ;
                            Rpp32f euclideanDistance = sqrt(newIsquared + newJsquared);

                            if (euclideanDistance >= 0 && euclideanDistance <= 1)
                            {
                                Rpp32f newEuclideanDistance = sqrt(1.0 - (euclideanDistance * euclideanDistance));
                                newEuclideanDistance = (euclideanDistance + (1.0 - newEuclideanDistance)) / 2.0;
                                if (newEuclideanDistance <= 1.0)
                                {
                                    Rpp32f theta = atan2(newI, newJ);

                                    Rpp32f newIsrc = newEuclideanDistance * sin(theta);
                                    Rpp32f newJsrc = newEuclideanDistance * cos(theta);

                                    int iSrc = (int) (((newIsrc + 1.0) * ((Rpp32f) batch_srcSize[batchCount].height)) / 2.0);
                                    int jSrc = (int) (((newJsrc + 1.0) * ((Rpp32f) batch_srcSize[batchCount].width)) / 2.0);

                                    int srcPosition = (int)(channel * ((iSrc * batch_srcSize[batchCount].width) + jSrc));

                                    if ((srcPosition >= 0) && (srcPosition < (channel * imageDim)))
                                    {
                                        int srcPositionTrue = (int)(channel * ((iSrc * batch_srcSizeMax[batchCount].width) + jSrc));
                                        for(int c = 0; c < channel; c++)
                                        {
                                            *dstPtrTemp = *(srcPtrTemp + c + srcPositionTrue);

                                            dstPtrTemp++;
                                        }
                                    }
                                    else
                                    {
                                        memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                                        dstPtrTemp += channel;
                                    }
                                }
                            }
                            else
                            {
                                dstPtrTemp += channel;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus fisheye_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f newI, newJ, newIsrc, newJsrc, newIsquared, newJsquared, euclideanDistance, newEuclideanDistance, theta;
    int iSrc, jSrc, srcPosition;
    Rpp32u elementsPerChannel = srcSize.height * srcSize.width;
    Rpp32u elements = channel * elementsPerChannel;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for(int i = 0; i < srcSize.height; i++)
            {
                newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(srcSize.height))) - 1.0;
                newIsquared = newI * newI;
                for(int j = 0; j < srcSize.width; j++)
                {
                    newJ = (((Rpp32f) (j * 2.0)) / ((Rpp32f)(srcSize.width))) - 1.0;
                    newJsquared = newJ * newJ;
                    euclideanDistance = sqrt(newIsquared + newJsquared);
                    if (euclideanDistance >= 0 && euclideanDistance <= 1)
                    {
                        newEuclideanDistance = sqrt(1.0 - (euclideanDistance * euclideanDistance));
                        newEuclideanDistance = (euclideanDistance + (1.0 - newEuclideanDistance)) / 2.0;
                        if (newEuclideanDistance <= 1.0)
                        {
                            theta = atan2(newI, newJ);

                            newIsrc = newEuclideanDistance * sin(theta);
                            newJsrc = newEuclideanDistance * cos(theta);

                            iSrc = (int) (((newIsrc + 1.0) * ((Rpp32f) srcSize.height)) / 2.0);
                            jSrc = (int) (((newJsrc + 1.0) * ((Rpp32f) srcSize.width)) / 2.0);

                            srcPosition = (int)((iSrc * srcSize.width) + jSrc);

                            if ((srcPosition >= 0) && (srcPosition < elementsPerChannel))
                            {
                                *dstPtrTemp = *(srcPtrTemp + srcPosition);
                            }
                            else
                            {
                                *dstPtrTemp = (T) 0;
                            }
                        }
                    }
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0; i < srcSize.height; i++)
        {
            newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(srcSize.height))) - 1.0;
            newIsquared = newI * newI;
            for(int j = 0; j < srcSize.width; j++)
            {
                for(int c = 0; c < channel; c++)
                {
                    srcPtrTemp = srcPtr + c;
                    newJ = (((Rpp32f) (j * 2.0)) / ((Rpp32f)(srcSize.width))) - 1.0;
                    newJsquared = newJ * newJ;
                    euclideanDistance = sqrt(newIsquared + newJsquared);
                    if (euclideanDistance >= 0 && euclideanDistance <= 1)
                    {
                        newEuclideanDistance = sqrt(1.0 - (euclideanDistance * euclideanDistance));
                        newEuclideanDistance = (euclideanDistance + (1.0 - newEuclideanDistance)) / 2.0;
                        if (newEuclideanDistance <= 1.0)
                        {
                            theta = atan2(newI, newJ);

                            newIsrc = newEuclideanDistance * sin(theta);
                            newJsrc = newEuclideanDistance * cos(theta);

                            iSrc = (int) (((newIsrc + 1.0) * ((Rpp32f) srcSize.height)) / 2.0);
                            jSrc = (int) (((newJsrc + 1.0) * ((Rpp32f) srcSize.width)) / 2.0);

                            srcPosition = (int)(channel * ((iSrc * srcSize.width) + jSrc));

                            if ((srcPosition >= 0) && (srcPosition < elements))
                            {
                                *dstPtrTemp = *(srcPtrTemp + srcPosition);
                            }
                            else
                            {
                                *dstPtrTemp = (T) 0;
                            }
                        }
                    }
                    dstPtrTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** lens_correction ***************/

template <typename T>
RppStatus lens_correction_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                                     Rpp32f *batch_strength, Rpp32f *batch_zoom, RppiROI *roiPoints,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            Rpp32f strength = batch_strength[batchCount];
            Rpp32f zoom = batch_zoom[batchCount];

            Rpp32f halfHeight = ((Rpp32f) batch_srcSize[batchCount].height) / 2.0;
            Rpp32f halfWidth = ((Rpp32f) batch_srcSize[batchCount].width) / 2.0;

            if (strength == 0) strength = 0.000001;
            Rpp32f correctionRadius = sqrt(batch_srcSize[batchCount].height * batch_srcSize[batchCount].height + batch_srcSize[batchCount].width * batch_srcSize[batchCount].width) / strength;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32f pixel;
                    
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32f newI = i - halfHeight;
                        
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32f newJ = j - halfWidth;
                                Rpp32f euclideanDistance = sqrt(newI * newI + newJ * newJ);
                                Rpp32f correctedDistance = euclideanDistance / correctionRadius;
                                Rpp32f theta;

                                if(correctedDistance == 0)
                                {
                                    theta = 1;
                                }
                                else
                                {
                                    theta = atan(correctedDistance) / correctedDistance;
                                }

                                Rpp32f srcLocationRow = halfHeight + theta * newI * zoom;
                                Rpp32f srcLocationColumn = halfWidth + theta * newJ * zoom;
                                
                                if ((srcLocationRow >= 0) && (srcLocationColumn >= 0) && 
                                    (srcLocationRow < batch_srcSize[batchCount].height) && (srcLocationColumn < batch_srcSize[batchCount].width))
                                {
                                    Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                                    Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                    if (srcLocationRowFloor > (batch_srcSize[batchCount].height - 2))
                                    {
                                        srcLocationRowFloor = batch_srcSize[batchCount].height - 2;
                                    }

                                    T *srcPtrTopRow, *srcPtrBottomRow;
                                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                                    srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;

                                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                                    if (srcLocationColumnFloor > (batch_srcSizeMax[batchCount].width - 2))
                                    {
                                        srcLocationColumnFloor = batch_srcSizeMax[batchCount].width - 2;
                                    }

                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;
                                }
                                else
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                

                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            Rpp32f strength = batch_strength[batchCount];
            Rpp32f zoom = batch_zoom[batchCount];

            Rpp32f halfHeight = ((Rpp32f) batch_srcSize[batchCount].height) / 2.0;
            Rpp32f halfWidth = ((Rpp32f) batch_srcSize[batchCount].width) / 2.0;

            if (strength == 0) strength = 0.000001;
            Rpp32f correctionRadius = sqrt(batch_srcSize[batchCount].height * batch_srcSize[batchCount].height + batch_srcSize[batchCount].width * batch_srcSize[batchCount].width) / strength;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            
            // #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;
                Rpp32f newI = i - halfHeight;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, elementsInRow * sizeof(T));
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32f newJ = newJ = j - halfWidth;
                            Rpp32f euclideanDistance = sqrt(newI * newI + newJ * newJ);
                            Rpp32f correctedDistance = euclideanDistance / correctionRadius;

                            Rpp32f theta;
                            if(correctedDistance == 0)
                            {
                                theta = 1;
                            }
                            else
                            {
                                theta = atan(correctedDistance) / correctedDistance;
                            }

                            Rpp32f srcLocationRow = halfHeight + theta * newI * zoom;
                            Rpp32f srcLocationColumn = halfWidth + theta * newJ * zoom;

                            if ((srcLocationRow >= 0) && (srcLocationColumn >= 0) && 
                                (srcLocationRow < batch_srcSize[batchCount].height) && (srcLocationColumn < batch_srcSize[batchCount].width))
                            {
                                Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                if (srcLocationRowFloor > (batch_srcSize[batchCount].height - 2))
                                {
                                    srcLocationRowFloor = batch_srcSize[batchCount].height - 2;
                                }

                                T *srcPtrTopRow, *srcPtrBottomRow;
                                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRowMax;
                                srcPtrBottomRow  = srcPtrTopRow + elementsInRowMax;

                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                                if (srcLocationColumnFloor > (batch_srcSize[batchCount].width - 2))
                                {
                                    srcLocationColumnFloor = batch_srcSize[batchCount].width - 2;
                                }

                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                for(int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                                    
                                    *dstPtrTemp = (T) pixel;

                                    dstPtrTemp++;
                                }
                            }
                            else
                            {
                                memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                                dstPtrTemp += channel;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus lens_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                               Rpp32f strength, Rpp32f zoom, 
                               RppiChnFormat chnFormat, Rpp32u channel)
{
    if (strength < 0)
    {
        return RPP_ERROR;
    }

    if (zoom < 1)
    {
        return RPP_ERROR;
    }
    
    Rpp32f halfHeight, halfWidth, newI, newJ, correctionRadius, euclideanDistance, correctedDistance, theta;
    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    halfHeight = ((Rpp32f) srcSize.height) / 2.0;
    halfWidth = ((Rpp32f) srcSize.width) / 2.0;

    if (strength == 0) strength = 0.000001;

    correctionRadius = sqrt(srcSize.height * srcSize.height + srcSize.width * srcSize.width) / strength;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSize.height; i++)
            {
                newI = i - halfHeight;
                for (int j = 0; j < srcSize.width; j++)
                {
                    newJ = j - halfWidth;

                    euclideanDistance = sqrt(newI * newI + newJ * newJ);
                    
                    correctedDistance = euclideanDistance / correctionRadius;

                    if(correctedDistance == 0)
                    {
                        theta = 1;
                    }
                    else
                    {
                        theta = atan(correctedDistance) / correctedDistance;
                    }

                    srcLocationRow = halfHeight + theta * newI * zoom;
                    srcLocationColumn = halfWidth + theta * newJ * zoom;
                    
                    if ((srcLocationRow >= 0) && (srcLocationColumn >= 0) && 
                        (srcLocationRow < srcSize.height) && (srcLocationColumn < srcSize.width))
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        if (srcLocationRowFloor > (srcSize.height - 2))
                        {
                            srcLocationRowFloor = srcSize.height - 2;
                        }

                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        if (srcLocationColumnFloor > (srcSize.width - 2))
                        {
                            srcLocationColumnFloor = srcSize.width - 2;
                        }

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                    }
                    else
                    {
                        *dstPtrTemp = (T) 0;
                    }
                    
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for(int c = 0; c < channel; c++)
                {
                    newI = i - halfHeight;
                    newJ = j - halfWidth;

                    euclideanDistance = sqrt(newI * newI + newJ * newJ);
                    
                    correctedDistance = euclideanDistance / correctionRadius;

                    if(correctedDistance == 0)
                    {
                        theta = 1;
                    }
                    else
                    {
                        theta = atan(correctedDistance) / correctedDistance;
                    }

                    srcLocationRow = halfHeight + theta * newI * zoom;
                    srcLocationColumn = halfWidth + theta * newJ * zoom;

                    if ((srcLocationRow >= 0) && (srcLocationColumn >= 0) && 
                        (srcLocationRow < srcSize.height) && (srcLocationColumn < srcSize.width))
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        if (srcLocationRowFloor > (srcSize.height - 2))
                        {
                            srcLocationRowFloor = srcSize.height - 2;
                        }

                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                        srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        if (srcLocationColumnFloor > (srcSize.width - 2))
                        {
                            srcLocationColumnFloor = srcSize.width - 2;
                        }

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                        
                        *dstPtrTemp = (T) pixel;
                    }
                    else
                    {
                        *dstPtrTemp = (T) 0;
                    }

                    dstPtrTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** scale ***************/

template <typename T>
RppStatus scale_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                             Rpp32f *batch_percentage, RppiROI *roiPoints,
                             Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f percentage = batch_percentage[batchCount] / 100;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);

                // #pragma omp parallel for
                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f pixel;
                    
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_dstSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_dstSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32f srcLocationRow = ((Rpp32f) i) / percentage;
                        Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                        T *srcPtrTopRow, *srcPtrBottomRow;
                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                        srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;
                        
                        for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32f srcLocationColumn = ((Rpp32f) j) / percentage;
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                                if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else
                                {
                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));
                                    
                                    *dstPtrTemp = (T) pixel;
                                }
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }
            
            Rpp32f percentage = batch_percentage[batchCount] / 100;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            
            Rpp32u dstElementsInRow = channel * batch_dstSize[batchCount].width;
            
            // #pragma omp parallel for
            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                Rpp32f pixel;
                
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, dstElementsInRow * sizeof(T));
                }
                else
                {
                    Rpp32f srcLocationRow = ((Rpp32f) i) / percentage;
                    Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                    T *srcPtrTopRow, *srcPtrBottomRow;
                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcElementsInRowMax;
                    srcPtrBottomRow  = srcPtrTopRow + srcElementsInRowMax;

                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32f srcLocationColumn = ((Rpp32f) j) / percentage;
                            Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                            Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                            if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else
                            {
                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;
                                
                                for (int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                                    
                                    *dstPtrTemp = (T) pixel;

                                    dstPtrTemp ++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus scale_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f percentage,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    if (dstSize.height < 0 || dstSize.width < 0)
    {
        return RPP_ERROR;
    }
    if (percentage < 0)
    {
        return RPP_ERROR;
    }

    percentage /= 100;

    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {   
                srcLocationRow = ((Rpp32f) i) / percentage;
                srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                srcPtrBottomRow  = srcPtrTopRow + srcSize.width;
                
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / percentage;
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = (T) 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));
                        
                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / percentage;
            srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            for (int j = 0; j < dstSize.width; j++)
            {   
                srcLocationColumn = ((Rpp32f) j) / percentage;
                srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                }
                else
                {
                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;
                    
                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                                + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                        
                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** rotate ***************/

template <typename T>
RppStatus rotate_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                             Rpp32f *batch_angleDeg, RppiROI *roiPoints, 
                             Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }
            
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f angleDeg = batch_angleDeg[batchCount];
            Rpp32f angleRad = -RAD(angleDeg);
            Rpp32f rotate[4] = {0};
            rotate[0] = cos(angleRad);
            rotate[1] = sin(angleRad);
            rotate[2] = -sin(angleRad);
            rotate[3] = cos(angleRad);
            Rpp32f divisor = (rotate[1] * rotate[2]) - (rotate[0] * rotate[3]);

            Rpp32f halfSrcHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfSrcWidth = batch_srcSize[batchCount].width / 2;
            Rpp32f halfDstHeight = batch_dstSize[batchCount].height / 2;
            Rpp32f halfDstWidth = batch_dstSize[batchCount].width / 2;
            Rpp32f halfHeightDiff = halfSrcHeight - halfDstHeight;
            Rpp32f halfWidthDiff = halfSrcWidth - halfDstWidth;

            Rpp32f srcLocationRowParameter = (rotate[0] * halfSrcHeight) + (rotate[1] * halfSrcWidth) - halfSrcHeight + halfHeightDiff;
            Rpp32f srcLocationColumnParameter = (rotate[2] * halfSrcHeight) + (rotate[3] * halfSrcWidth) - halfSrcWidth + halfWidthDiff;
            Rpp32f srcLocationRowParameter2 = (-rotate[3] * (Rpp32s)srcLocationRowParameter) + (rotate[1] * (Rpp32s)srcLocationColumnParameter);
            Rpp32f srcLocationColumnParameter2 = (rotate[2] * (Rpp32s)srcLocationRowParameter) + (-rotate[0] * (Rpp32s)srcLocationColumnParameter);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);

                // #pragma omp parallel for
                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f pixel;
                    
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_dstSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_dstSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32f srcLocationRowTerm1 = -rotate[3] * i;
                        Rpp32f srcLocationColumnTerm1 = rotate[2] * i;
                        
                        for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32f srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) / divisor;
                                Rpp32f srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) / divisor;
                                
                                Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                                if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else
                                {
                                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                                    
                                    T *srcPtrTopRow, *srcPtrBottomRow;
                                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                                    srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;

                                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));
                                
                                    *dstPtrTemp = (T) pixel;
                                }
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32f angleDeg = batch_angleDeg[batchCount];
            Rpp32f angleRad = -RAD(angleDeg);
            Rpp32f rotate[4] = {0};
            rotate[0] = cos(angleRad);
            rotate[1] = sin(angleRad);
            rotate[2] = -sin(angleRad);
            rotate[3] = cos(angleRad);
            Rpp32f divisor = (rotate[1] * rotate[2]) - (rotate[0] * rotate[3]);

            Rpp32f halfSrcHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfSrcWidth = batch_srcSize[batchCount].width / 2;
            Rpp32f halfDstHeight = batch_dstSize[batchCount].height / 2;
            Rpp32f halfDstWidth = batch_dstSize[batchCount].width / 2;
            Rpp32f halfHeightDiff = halfSrcHeight - halfDstHeight;
            Rpp32f halfWidthDiff = halfSrcWidth - halfDstWidth;

            Rpp32f srcLocationRowParameter = (rotate[0] * halfSrcHeight) + (rotate[1] * halfSrcWidth) - halfSrcHeight + halfHeightDiff;
            Rpp32f srcLocationColumnParameter = (rotate[2] * halfSrcHeight) + (rotate[3] * halfSrcWidth) - halfSrcWidth + halfWidthDiff;
            Rpp32f srcLocationRowParameter2 = (-rotate[3] * (Rpp32s)srcLocationRowParameter) + (rotate[1] * (Rpp32s)srcLocationColumnParameter);
            Rpp32f srcLocationColumnParameter2 = (rotate[2] * (Rpp32s)srcLocationRowParameter) + (-rotate[0] * (Rpp32s)srcLocationColumnParameter);

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;

            Rpp32u dstElementsInRow = channel * batch_dstSize[batchCount].width;
            
            // #pragma omp parallel for
            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                Rpp32f pixel;
                
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, dstElementsInRow * sizeof(T));
                }
                else
                {
                    Rpp32f srcLocationRowTerm1 = -rotate[3] * i;
                    Rpp32f srcLocationColumnTerm1 = rotate[2] * i;

                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32f srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) / divisor;
                            Rpp32f srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) / divisor;

                            Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                            Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                            if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                            {
                                memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                                dstPtrTemp += channel;
                            }
                            else
                            {
                                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                                
                                T *srcPtrTopRow, *srcPtrBottomRow;
                                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcElementsInRowMax;
                                srcPtrBottomRow  = srcPtrTopRow + srcElementsInRowMax;

                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                for (int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                                
                                    *dstPtrTemp = (T) pixel;
                                    dstPtrTemp ++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus rotate_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f angleDeg,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f angleRad = -RAD(angleDeg);
    Rpp32f rotate[4] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = -sin(angleRad);
    rotate[3] = cos(angleRad);

    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    
    Rpp32f divisor = (rotate[1] * rotate[2]) - (rotate[0] * rotate[3]);
    Rpp32f srcLocationRow, srcLocationColumn, srcLocationRowTerm1, srcLocationColumnTerm1, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;

    Rpp32f halfSrcHeight = srcSize.height / 2;
    Rpp32f halfSrcWidth = srcSize.width / 2;
    Rpp32f halfDstHeight = dstSize.height / 2;
    Rpp32f halfDstWidth = dstSize.width / 2;
    Rpp32f halfHeightDiff = halfSrcHeight - halfDstHeight;
    Rpp32f halfWidthDiff = halfSrcWidth - halfDstWidth;

    Rpp32f srcLocationRowParameter = (rotate[0] * halfSrcHeight) + (rotate[1] * halfSrcWidth) - halfSrcHeight + halfHeightDiff;
    Rpp32f srcLocationColumnParameter = (rotate[2] * halfSrcHeight) + (rotate[3] * halfSrcWidth) - halfSrcWidth + halfWidthDiff;
    Rpp32f srcLocationRowParameter2 = (-rotate[3] * (Rpp32s)srcLocationRowParameter) + (rotate[1] * (Rpp32s)srcLocationColumnParameter);
    Rpp32f srcLocationColumnParameter2 = (rotate[2] * (Rpp32s)srcLocationRowParameter) + (-rotate[0] * (Rpp32s)srcLocationColumnParameter);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                srcLocationRowTerm1 = -rotate[3] * i;
                srcLocationColumnTerm1 = rotate[2] * i;
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) / divisor;
                    srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) / divisor;
                    
                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        
                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));
                    
                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRowTerm1 = -rotate[3] * i;
            srcLocationColumnTerm1 = rotate[2] * i;
            for (int j = 0; j < dstSize.width; j++)
            {
                srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter2) / divisor;
                srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter2) / divisor;
                
                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                    dstPtrTemp += channel;
                }
                else
                {
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                    
                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                    
                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** resize ***************/

template <typename T>
RppStatus resize_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                            RppiROI *roiPoints, Rpp32u nbatchSize,
                            RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;
            
            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;
            
            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

            compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = roiPoints[batchCount].roiHeight;
            srcSizeROI.width = roiPoints[batchCount].roiWidth;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;
            
            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

            compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus resize_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    resize_kernel_host(srcPtr, srcSize, dstPtr, dstSize, chnFormat, channel);

    return RPP_SUCCESS;
}

// /**************** resize_crop ***************/

template <typename T>
RppStatus resize_crop_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                                 Rpp32u *batch_x1, Rpp32u *batch_x2, Rpp32u *batch_y1, Rpp32u *batch_y2,
                                 Rpp32u nbatchSize,
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;
            
            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;
            
            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

            compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;
            
            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

            compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus resize_crop_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32u x1, Rpp32u x2, Rpp32u y1, Rpp32u y2,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    resize_crop_kernel_host(srcPtr, srcSize, dstPtr, dstSize, x1, y1, x2, y2, chnFormat, channel);

    return RPP_SUCCESS;
    
}

/**************** warp_affine ***************/

template <typename T>
RppStatus warp_affine_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, RppiROI *roiPoints,
                                 Rpp32f *batch_affine,
                                 Rpp32u nbatchSize,
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    //Rpp32f affine[6] = {1.35, 0.3, 0, -0.75, 1.1, 0};
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f affine[6] = {0};
            for (int i = 0; i < 6; i++)
            {
                affine[i] = batch_affine[(batchCount * 6) + i];
            }
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);

                // #pragma omp parallel for
                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f pixel;
                    
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_dstSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_dstSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32s iNew = i - (batch_srcSize[batchCount].height / 2);
                        
                        for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32s jNew = j - (batch_srcSize[batchCount].width / 2);
                                
                                Rpp32f srcLocationColumn = (jNew * affine[0]) + (iNew * affine[1]) + affine[2];
                                Rpp32f srcLocationRow = (jNew * affine[3]) + (iNew * affine[4]) + affine[5];

                                srcLocationColumn += (batch_srcSize[batchCount].width / 2);
                                srcLocationRow += (batch_srcSize[batchCount].height / 2);

                                Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                                if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                                {
                                    *dstPtrTemp = 0;
                                }
                                else
                                {
                                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                                    
                                    T *srcPtrTopRow, *srcPtrBottomRow;
                                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                                    srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;

                                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;
                                }
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }
            
            Rpp32f affine[6] = {0};
            for (int i = 0; i < 6; i++)
            {
                affine[i] = batch_affine[(batchCount * 6) + i];
            }
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;

            Rpp32u dstElementsInRow = channel * batch_dstSize[batchCount].width;
            
            // #pragma omp parallel for
            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                Rpp32f pixel;
                
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, dstElementsInRow * sizeof(T));
                }
                else
                {
                    Rpp32s iNew = i - (batch_srcSize[batchCount].height / 2);

                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32s jNew = j - (batch_srcSize[batchCount].width / 2);
                            
                            Rpp32f srcLocationColumn = (jNew * affine[0]) + (iNew * affine[1]) + affine[2];
                            Rpp32f srcLocationRow = (jNew * affine[3]) + (iNew * affine[4]) + affine[5];

                            srcLocationColumn += (batch_srcSize[batchCount].width / 2);
                            srcLocationRow += (batch_srcSize[batchCount].height / 2);

                            Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                            Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                            if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                            {
                                memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                                dstPtrTemp += channel;
                            }
                            else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                            {
                                memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                                dstPtrTemp += channel;
                            }
                            else
                            {
                                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                                
                                T *srcPtrTopRow, *srcPtrBottomRow;
                                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcElementsInRowMax;
                                srcPtrBottomRow  = srcPtrTopRow + srcElementsInRowMax;

                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                for (int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;

                                    dstPtrTemp ++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus warp_affine_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f* affine,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    Rpp32f srcLocationRow, srcLocationColumn, pixel;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                Rpp32s iNew = i - (srcSize.height / 2);
                for (int j = 0; j < dstSize.width; j++)
                {
                    Rpp32s jNew = j - (srcSize.width / 2);
                    
                    srcLocationColumn = (jNew * affine[0]) + (iNew * affine[1]) + affine[2];
                    srcLocationRow = (jNew * affine[3]) + (iNew * affine[4]) + affine[5];

                    srcLocationColumn += (srcSize.width / 2);
                    srcLocationRow += (srcSize.height / 2);

                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        
                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }

                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            Rpp32s iNew = i - (srcSize.height / 2);
            for (int j = 0; j < dstSize.width; j++)
            {
                Rpp32s jNew = j - (srcSize.width / 2);

                srcLocationColumn = (jNew * affine[0]) + (iNew * affine[1]) + affine[2];
                srcLocationRow = (jNew * affine[3]) + (iNew * affine[4]) + affine[5];

                srcLocationColumn += (srcSize.width / 2);
                srcLocationRow += (srcSize.height / 2);

                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    memset(dstPtrTemp, (T) 0, channel * sizeof(T));
                    dstPtrTemp += channel;
                }
                else
                {
                    Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                    
                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** warp_perspective ***************/

template <typename T>
RppStatus warp_perspective_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                                      RppiROI *roiPoints, Rpp32f *batch_perspective,
                                      Rpp32u nbatchSize,
                                      RppiChnFormat chnFormat, Rpp32u channel)
{
    //Rpp32f perspective[9] = {0.707, 0.707, 0, -0.707, 0.707, 0, 0.001, 0.001, 1};
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }
            
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f perspective[9] = {0};
            for (int i = 0; i < 9; i++)
            {
                perspective[i] = batch_perspective[(batchCount * 9) + i];
            }
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcImageDimMax);
                dstPtrChannel = dstPtrImage + (c * dstImageDimMax);

                // #pragma omp parallel for
                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f pixel;
                    
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel;
                    dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memset(dstPtrTemp, (T) 0, batch_dstSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_dstSize[batchCount].width;
                    }
                    else
                    {
                        Rpp32s iNew = i - (batch_srcSize[batchCount].height / 2);
                        
                        for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            if (!((x1 <= j) && (j <= x2 )))
                            {
                                memset(dstPtrTemp, (T) 0, sizeof(T));

                                dstPtrTemp += 1;
                            }
                            else
                            {
                                Rpp32s jNew = j - (batch_srcSize[batchCount].width / 2);
                                
                                Rpp32f srcLocationColumn = ((jNew * perspective[0]) + (iNew * perspective[1]) + perspective[2]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);
                                Rpp32f srcLocationRow = ((jNew * perspective[3]) + (iNew * perspective[4]) + perspective[5]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);

                                srcLocationColumn += (batch_srcSize[batchCount].width / 2);
                                srcLocationRow += (batch_srcSize[batchCount].height / 2);

                                Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                                Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                                if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                                {
                                    *dstPtrTemp = (T) 0;
                                }
                                else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                                {
                                    *dstPtrTemp = 0;
                                }
                                else
                                {
                                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                                    
                                    T *srcPtrTopRow, *srcPtrBottomRow;
                                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * batch_srcSizeMax[batchCount].width;
                                    srcPtrBottomRow  = srcPtrTopRow + batch_srcSizeMax[batchCount].width;

                                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;
                                }
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32f perspective[9] = {0};
            for (int i = 0; i < 9; i++)
            {
                perspective[i] = batch_perspective[(batchCount * 9) + i];
            }
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;

            Rpp32u dstElementsInRow = channel * batch_dstSizeMax[batchCount].width;
            
            // #pragma omp parallel for
            for(int i = 0; i < batch_dstSize[batchCount].height; i++)
            {
                Rpp32f pixel;
                
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memset(dstPtrTemp, (T) 0, dstElementsInRow * sizeof(T));
                }
                else
                {
                    Rpp32s iNew = i - (batch_srcSize[batchCount].height / 2);

                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memset(dstPtrTemp, (T) 0, channel * sizeof(T));

                            dstPtrTemp += channel;
                        }
                        else
                        {
                            Rpp32s jNew = j - (batch_srcSize[batchCount].width / 2);
                            
                            Rpp32f srcLocationColumn = ((jNew * perspective[0]) + (iNew * perspective[1]) + perspective[2]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);
                            Rpp32f srcLocationRow = ((jNew * perspective[3]) + (iNew * perspective[4]) + perspective[5]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);

                            srcLocationColumn += (batch_srcSize[batchCount].width / 2);
                            srcLocationRow += (batch_srcSize[batchCount].height / 2);

                            Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                            Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                            if ((srcLocationRowFloor < y1) || (srcLocationRowFloor > y2) || (srcLocationColumnFloor < x1) || (srcLocationColumnFloor > x2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (batch_srcSize[batchCount].height - 2) || srcLocationColumn > (batch_srcSize[batchCount].width - 2))
                            {
                                for (int c = 0; c < channel; c++)
                                {
                                    *dstPtrTemp = 0;

                                    dstPtrTemp++;
                                }
                            }
                            else
                            {
                                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                                
                                T *srcPtrTopRow, *srcPtrBottomRow;
                                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcElementsInRowMax;
                                srcPtrBottomRow  = srcPtrTopRow + srcElementsInRowMax;

                                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                                for (int c = 0; c < channel; c++)
                                {
                                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                                        + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                                    *dstPtrTemp = (T) pixel;

                                    dstPtrTemp ++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus warp_perspective_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f* perspective,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    Rpp32f srcLocationRow, srcLocationColumn, pixel;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                Rpp32s iNew = i - (srcSize.height / 2);
                for (int j = 0; j < dstSize.width; j++)
                {
                    Rpp32s jNew = j - (srcSize.width / 2);
                    
                    srcLocationColumn = ((jNew * perspective[0]) + (iNew * perspective[1]) + perspective[2]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);
                    srcLocationRow = ((jNew * perspective[3]) + (iNew * perspective[4]) + perspective[5]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);

                    srcLocationColumn += (srcSize.width / 2);
                    srcLocationRow += (srcSize.height / 2);

                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        
                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }

                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            Rpp32s iNew = i - (srcSize.height / 2);
            for (int j = 0; j < dstSize.width; j++)
            {
                Rpp32s jNew = j - (srcSize.width / 2);

                srcLocationColumn = ((jNew * perspective[0]) + (iNew * perspective[1]) + perspective[2]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);
                srcLocationRow = ((jNew * perspective[3]) + (iNew * perspective[4]) + perspective[5]) / ((jNew * perspective[6]) + (iNew * perspective[7]) + perspective[8]);

                srcLocationColumn += (srcSize.width / 2);
                srcLocationRow += (srcSize.height / 2);

                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                }
                else
                {
                    Rpp32s srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32s srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                    
                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (T) pixel;
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}