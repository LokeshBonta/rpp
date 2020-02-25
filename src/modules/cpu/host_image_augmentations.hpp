#include "cpu/rpp_cpu_simd.hpp"
#include <cpu/rpp_cpu_common.hpp>
#include <stdlib.h>
#include <time.h>

// #if ENABLE_SIMD_INTRINSICS

#if 1

static unsigned int g_seed;
//Used to seed the generator.
inline void fast_srand( int seed )
{
    g_seed = seed;
}

//fastrand routine returns one integer, similar output value range as C lib. :: taken from intel
inline int fastrand()
{
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}

/************ brightness ************/

template <typename T>
RppStatus brightness_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                                Rpp32f *batch_alpha, Rpp32f *batch_beta, 
                                RppiROI *roiPoints, Rpp32u nbatchSize,
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

            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];
            
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
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                        Rpp32u alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128 pMul = _mm_set1_ps(alpha);
                        __m128 pAdd = _mm_set1_ps(beta);
                        __m128 p0, p1, p2, p3;
                        __m128i px0, px1, px2, px3;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                            
                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
                            
                            p0 = _mm_mul_ps(p0, pMul);
                            p1 = _mm_mul_ps(p1, pMul);
                            p2 = _mm_mul_ps(p2, pMul);
                            p3 = _mm_mul_ps(p3, pMul);
                            px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
                            px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
                            px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
                            px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));
                            
                            px0 = _mm_packus_epi32(px0, px1);
                            px1 = _mm_packus_epi32(px2, px3);
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
                            
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            
                            srcPtrTemp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
                        }

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
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
            
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];
            
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

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;
                    Rpp32u alignedLength = bufferLength & ~15;

                    __m128i const zero = _mm_setzero_si128();
                    __m128 pMul = _mm_set1_ps(alpha);
                    __m128 pAdd = _mm_set1_ps(beta);
                    __m128 p0, p1, p2, p3;
                    __m128i px0, px1, px2, px3;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                        
                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
                        
                        p0 = _mm_mul_ps(p0, pMul);
                        p1 = _mm_mul_ps(p1, pMul);
                        p2 = _mm_mul_ps(p2, pMul);
                        p3 = _mm_mul_ps(p3, pMul);
                        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
                        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
                        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
                        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));
                        
                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
                        
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        
                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
                    }

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus brightness_host(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,
                          Rpp32f alpha, Rpp32f beta,
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;

    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    
    int bufferLength = channel * srcSize.height * srcSize.width;
    int alignedLength = bufferLength & ~15;

    __m128i const zero = _mm_setzero_si128();
    __m128 pMul = _mm_set1_ps(alpha), pAdd = _mm_set1_ps(beta);
    __m128 p0, p1, p2, p3;
    __m128i px0, px1, px2, px3;

    int vectorLoopCount = 0;
    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
    {
        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
        
        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
        
        p0 = _mm_mul_ps(p0, pMul);
        p1 = _mm_mul_ps(p1, pMul);
        p2 = _mm_mul_ps(p2, pMul);
        p3 = _mm_mul_ps(p3, pMul);
        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));
        
        px0 = _mm_packus_epi32(px0, px1);
        px1 = _mm_packus_epi32(px2, px3);
        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
        
        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
        
        srcPtrTemp +=16;
        dstPtrTemp +=16;
    }
    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
    {
        *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
    }

    return RPP_SUCCESS;
}

// /**************** contrast ***************/

template <typename T>
RppStatus contrast_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                              Rpp32u *batch_new_min, Rpp32u *batch_new_max, 
                              RppiROI *roiPoints, Rpp32u nbatchSize,
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

            Rpp32f new_min = batch_new_min[batchCount];
            Rpp32f new_max = batch_new_max[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            T min[3] = {255, 255, 255};
            T max[3] = {0, 0, 0};

            if (channel == 1)
            {
                compute_1_channel_minmax_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], &min[0], &max[0], chnFormat, channel);   
            }
            else if (channel == 3)
            {
                compute_3_channel_minmax_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], min, max, chnFormat, channel);
            }

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                Rpp32f contrastFactor = (Rpp32f) (new_max - new_min) / (max[c] - min[c]);

                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                        Rpp32u alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128 pContrastFactor = _mm_set1_ps(contrastFactor);
                        __m128i pMin = _mm_set1_epi16(min[c]);
                        __m128i pNewMin = _mm_set1_epi16(new_min);
                        __m128 p0, p1, p2, p3;
                        __m128i px0, px1;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                            px1 = _mm_sub_epi16(_mm_unpackhi_epi8(px0, zero), pMin);    // pixels 8-15
                            px0 = _mm_sub_epi16(_mm_unpacklo_epi8(px0, zero), pMin);    // pixels 0-7
                            p2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
                            p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11

                            p0 = _mm_mul_ps(p0, pContrastFactor);
                            p2 = _mm_mul_ps(p2, pContrastFactor);
                            p1 = _mm_mul_ps(p1, pContrastFactor);
                            p3 = _mm_mul_ps(p3, pContrastFactor);

                            px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p2));
                            px1 = _mm_packus_epi32(_mm_cvtps_epi32(p1), _mm_cvtps_epi32(p3));
                            
                            px0 = _mm_add_epi16(px0, pNewMin);
                            px1 = _mm_add_epi16(px1, pNewMin);
                            
                            _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));
                            
                            srcPtrTemp += 16;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[c]) * contrastFactor) + new_min);
                        }

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
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

            Rpp32f new_min = batch_new_min[batchCount];
            Rpp32f new_max = batch_new_max[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            T min[3] = {255, 255, 255};
            T max[3] = {0, 0, 0};

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            compute_3_channel_minmax_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], min, max, chnFormat, channel);

            Rpp32f contrastFactorR = (Rpp32f) (new_max - new_min) / (max[0] - min[0]);
            Rpp32f contrastFactorG = (Rpp32f) (new_max - new_min) / (max[1] - min[1]);
            Rpp32f contrastFactorB = (Rpp32f) (new_max - new_min) / (max[2] - min[2]);

            // #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;

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

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;
                    Rpp32u alignedLength = bufferLength & ~14;

                    __m128i const zero = _mm_setzero_si128();
                    __m128 pContrastFactor0 = _mm_set_ps(contrastFactorR, contrastFactorG, contrastFactorB, contrastFactorR);
                    __m128 pContrastFactor1 = _mm_set_ps(contrastFactorG, contrastFactorB, contrastFactorR, contrastFactorG);
                    __m128 pContrastFactor2 = _mm_set_ps(contrastFactorB, contrastFactorR, contrastFactorG, contrastFactorB);
                    
                    __m128i pMin0 = _mm_set_epi16(min[0], min[1], min[2], min[0], min[1], min[2], min[0], min[1]);
                    __m128i pMin1 = _mm_set_epi16(min[2], min[0], min[1], min[2], min[0], min[1], min[2], min[0]);
                    __m128i pNewMin = _mm_set1_epi16(new_min);
                    __m128 p0, p1, p2, p3;
                    __m128i px0, px1;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                        px1 = _mm_sub_epi16(_mm_unpackhi_epi8(px0, zero), pMin1);    // pixels 8-15
                        px0 = _mm_sub_epi16(_mm_unpacklo_epi8(px0, zero), pMin0);    // pixels 0-7
                        p2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
                        p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11

                        p0 = _mm_mul_ps(p0, pContrastFactor0);
                        p2 = _mm_mul_ps(p2, pContrastFactor1);
                        p1 = _mm_mul_ps(p1, pContrastFactor2);
                        p3 = _mm_mul_ps(p3, pContrastFactor0);

                        px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p2));
                        px1 = _mm_packus_epi32(_mm_cvtps_epi32(p1), _mm_cvtps_epi32(p3));
                        
                        px0 = _mm_add_epi16(px0, pNewMin);
                        px1 = _mm_add_epi16(px1, pNewMin);
                        
                        _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));
                        
                        srcPtrTemp += 15;
                        dstPtrTemp += 15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[0]) * contrastFactorR) + new_min);
                        *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[1]) * contrastFactorG) + new_min);
                        *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[2]) * contrastFactorB) + new_min);
                    }

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus contrast_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        Rpp32u new_min, Rpp32u new_max,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    T min[3] = {255, 255, 255};
    T max[3] = {0, 0, 0};

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (channel == 1)
        {
            compute_1_channel_minmax_host(srcPtr, srcSize, srcSize, &min[0], &max[0], chnFormat, channel);   
        }
        else if (channel == 3)
        {
            compute_3_channel_minmax_host(srcPtr, srcSize, srcSize, min, max, chnFormat, channel);
        }

        for (int c = 0; c < channel; c++)
        {
            Rpp32f contrastFactor = (Rpp32f) (new_max - new_min) / (max[c] - min[c]);

            int bufferLength = srcSize.height * srcSize.width;
            int alignedLength = bufferLength & ~15;

            __m128i const zero = _mm_setzero_si128();
            __m128 pContrastFactor = _mm_set1_ps(contrastFactor);
            __m128i pMin = _mm_set1_epi16(min[c]);
            __m128i pNewMin = _mm_set1_epi16(new_min);
            __m128 p0, p1, p2, p3;
            __m128i px0, px1;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                px1 = _mm_sub_epi16(_mm_unpackhi_epi8(px0, zero), pMin);    // pixels 8-15
                px0 = _mm_sub_epi16(_mm_unpacklo_epi8(px0, zero), pMin);    // pixels 0-7
                p2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
                p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11

                p0 = _mm_mul_ps(p0, pContrastFactor);
                p2 = _mm_mul_ps(p2, pContrastFactor);
                p1 = _mm_mul_ps(p1, pContrastFactor);
                p3 = _mm_mul_ps(p3, pContrastFactor);

                px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p2));
                px1 = _mm_packus_epi32(_mm_cvtps_epi32(p1), _mm_cvtps_epi32(p3));
                
                px0 = _mm_add_epi16(px0, pNewMin);
                px1 = _mm_add_epi16(px1, pNewMin);
                
                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));
                
                srcPtrTemp += 16;
                dstPtrTemp += 16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[c]) * contrastFactor) + new_min);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        compute_3_channel_minmax_host(srcPtr, srcSize, srcSize, min, max, chnFormat, channel);

        Rpp32f contrastFactorR = (Rpp32f) (new_max - new_min) / (max[0] - min[0]);
        Rpp32f contrastFactorG = (Rpp32f) (new_max - new_min) / (max[1] - min[1]);
        Rpp32f contrastFactorB = (Rpp32f) (new_max - new_min) / (max[2] - min[2]);

        int bufferLength = channel * srcSize.height * srcSize.width;
        int alignedLength = bufferLength & ~14;

        __m128i const zero = _mm_setzero_si128();
        __m128 pContrastFactor0 = _mm_set_ps(contrastFactorR, contrastFactorG, contrastFactorB, contrastFactorR);
        __m128 pContrastFactor1 = _mm_set_ps(contrastFactorG, contrastFactorB, contrastFactorR, contrastFactorG);
        __m128 pContrastFactor2 = _mm_set_ps(contrastFactorB, contrastFactorR, contrastFactorG, contrastFactorB);
        
        __m128i pMin0 = _mm_set_epi16(min[0], min[1], min[2], min[0], min[1], min[2], min[0], min[1]);
        __m128i pMin1 = _mm_set_epi16(min[2], min[0], min[1], min[2], min[0], min[1], min[2], min[0]);
        __m128i pNewMin = _mm_set1_epi16(new_min);
        __m128 p0, p1, p2, p3;
        __m128i px0, px1;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

            px1 = _mm_sub_epi16(_mm_unpackhi_epi8(px0, zero), pMin1);    // pixels 8-15
            px0 = _mm_sub_epi16(_mm_unpacklo_epi8(px0, zero), pMin0);    // pixels 0-7
            p2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
            p1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11

            p0 = _mm_mul_ps(p0, pContrastFactor0);
            p2 = _mm_mul_ps(p2, pContrastFactor1);
            p1 = _mm_mul_ps(p1, pContrastFactor2);
            p3 = _mm_mul_ps(p3, pContrastFactor0);

            px0 = _mm_packus_epi32(_mm_cvtps_epi32(p0), _mm_cvtps_epi32(p2));
            px1 = _mm_packus_epi32(_mm_cvtps_epi32(p1), _mm_cvtps_epi32(p3));
            
            px0 = _mm_add_epi16(px0, pNewMin);
            px1 = _mm_add_epi16(px1, pNewMin);
            
            _mm_storeu_si128((__m128i *)dstPtrTemp, _mm_packus_epi16(px0,px1));
            
            srcPtrTemp += 15;
            dstPtrTemp += 15;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
        {
            *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[0]) * contrastFactorR) + new_min);
            *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[1]) * contrastFactorG) + new_min);
            *dstPtrTemp++ = (T) (((Rpp32f)(*srcPtrTemp++ - min[2]) * contrastFactorB) + new_min);
        }
    }

    return RPP_SUCCESS;
}

/************ blend ************/

template <typename T>
RppStatus blend_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                           Rpp32f *batch_alpha, 
                           RppiROI *roiPoints, Rpp32u nbatchSize,
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

            Rpp32f alpha = batch_alpha[batchCount];
            
            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
                srcPtr1Channel = srcPtr1Image + (c * imageDimMax);
                srcPtr2Channel = srcPtr2Image + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32f pixel;

                    T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Channel + (i * batch_srcSizeMax[batchCount].width);
                    srcPtr2Temp = srcPtr2Channel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtr1Temp, batch_srcSize[batchCount].width * sizeof(T));

                        srcPtr1Temp += batch_srcSizeMax[batchCount].width;
                        srcPtr2Temp += batch_srcSizeMax[batchCount].width;
                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtr1Temp, x1 * sizeof(T));
                        srcPtr1Temp += x1;
                        srcPtr2Temp += x1;
                        dstPtrTemp += x1;

                        int bufferLength = roiPoints[batchCount].roiWidth;
                        int alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128 pMul = _mm_set1_ps(alpha);
                        __m128 p0, p1, p2, p3;
                        __m128 q0, q1, q2, q3;
                        __m128i px0, px1, px2, px3;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtr1Temp);
                            
                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            px0 =  _mm_loadu_si128((__m128i *)srcPtr2Temp);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            q0 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero)), p0);    // pixels 0-3
                            q1 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero)), p1);    // pixels 4-7
                            q2 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero)), p2);    // pixels 8-11
                            q3 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero)), p3);    // pixels 12-15

                            q0 = _mm_mul_ps(q0, pMul);
                            q1 = _mm_mul_ps(q1, pMul);
                            q2 = _mm_mul_ps(q2, pMul);
                            q3 = _mm_mul_ps(q3, pMul);
                            px0 = _mm_cvtps_epi32(_mm_add_ps(p0, q0));
                            px1 = _mm_cvtps_epi32(_mm_add_ps(p1, q1));
                            px2 = _mm_cvtps_epi32(_mm_add_ps(p2, q2));
                            px3 = _mm_cvtps_epi32(_mm_add_ps(p3, q3));
                            
                            px0 = _mm_packus_epi32(px0, px1);
                            px1 = _mm_packus_epi32(px2, px3);
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
                            
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            
                            srcPtr1Temp +=16;
                            srcPtr2Temp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK(*srcPtr1Temp + (alpha * (*srcPtr2Temp - *srcPtr1Temp)));
                            dstPtrTemp++;
                            srcPtr2Temp++;
                            srcPtr1Temp++;
                        }

                        memcpy(dstPtrTemp, srcPtr1Temp, remainingElementsAfterROI * sizeof(T));
                        srcPtr1Temp += remainingElementsAfterROI;
                        srcPtr2Temp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
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
            
            Rpp32f alpha = batch_alpha[batchCount];
            
            T *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtr1Image = srcPtr1 + loc;
            srcPtr2Image = srcPtr2 + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            
            // #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;
                
                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMax);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtr1Temp, elementsInRow * sizeof(T));

                    srcPtr1Temp += elementsInRowMax;
                    srcPtr2Temp += elementsInRowMax;
                    dstPtrTemp += elementsInRowMax;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtr1Temp, elementsBeforeROI * sizeof(T));
                    srcPtr1Temp += elementsBeforeROI;
                    srcPtr2Temp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    int bufferLength = channel * roiPoints[batchCount].roiWidth;
                    int alignedLength = bufferLength & ~15;

                    __m128i const zero = _mm_setzero_si128();
                    __m128 pMul = _mm_set1_ps(alpha);
                    __m128 p0, p1, p2, p3;
                    __m128 q0, q1, q2, q3;
                    __m128i px0, px1, px2, px3;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtr1Temp);
                        
                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        px0 =  _mm_loadu_si128((__m128i *)srcPtr2Temp);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        q0 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero)), p0);    // pixels 0-3
                        q1 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero)), p1);    // pixels 4-7
                        q2 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero)), p2);    // pixels 8-11
                        q3 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero)), p3);    // pixels 12-15

                        q0 = _mm_mul_ps(q0, pMul);
                        q1 = _mm_mul_ps(q1, pMul);
                        q2 = _mm_mul_ps(q2, pMul);
                        q3 = _mm_mul_ps(q3, pMul);
                        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, q0));
                        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, q1));
                        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, q2));
                        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, q3));
                        
                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
                        
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        
                        srcPtr1Temp +=16;
                        srcPtr2Temp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(*srcPtr1Temp + (alpha * (*srcPtr2Temp - *srcPtr1Temp)));
                        dstPtrTemp++;
                        srcPtr2Temp++;
                        srcPtr1Temp++;
                    }

                    memcpy(dstPtrTemp, srcPtr1Temp, remainingElementsAfterROI * sizeof(T));
                    srcPtr1Temp += remainingElementsAfterROI;
                    srcPtr2Temp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus blend_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr, 
                        Rpp32f alpha, RppiChnFormat chnFormat, 
                        unsigned int channel)
{
    T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;

    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    int bufferLength = channel * srcSize.height * srcSize.width;
    int alignedLength = bufferLength & ~15;

    __m128i const zero = _mm_setzero_si128();
    __m128 pMul = _mm_set1_ps(alpha);
    __m128 p0, p1, p2, p3;
    __m128 q0, q1, q2, q3;
    __m128i px0, px1, px2, px3;

    int vectorLoopCount = 0;
    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
    {
        px0 =  _mm_loadu_si128((__m128i *)srcPtr1Temp);
        
        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

        px0 =  _mm_loadu_si128((__m128i *)srcPtr2Temp);

        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        q0 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero)), p0);    // pixels 0-3
        q1 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero)), p1);    // pixels 4-7
        q2 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero)), p2);    // pixels 8-11
        q3 = _mm_add_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero)), p3);    // pixels 12-15

        q0 = _mm_mul_ps(q0, pMul);
        q1 = _mm_mul_ps(q1, pMul);
        q2 = _mm_mul_ps(q2, pMul);
        q3 = _mm_mul_ps(q3, pMul);
        px0 = _mm_cvtps_epi32(_mm_add_ps(p0, q0));
        px1 = _mm_cvtps_epi32(_mm_add_ps(p1, q1));
        px2 = _mm_cvtps_epi32(_mm_add_ps(p2, q2));
        px3 = _mm_cvtps_epi32(_mm_add_ps(p3, q3));
        
        px0 = _mm_packus_epi32(px0, px1);
        px1 = _mm_packus_epi32(px2, px3);
        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
        
        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
        
        srcPtr1Temp +=16;
        srcPtr2Temp +=16;
        dstPtrTemp +=16;
    }
    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
    {
        *dstPtrTemp = (T) RPPPIXELCHECK(*srcPtr1Temp + (alpha * (*srcPtr2Temp - *srcPtr1Temp)));
        dstPtrTemp++;
        srcPtr2Temp++;
        srcPtr1Temp++;
    }
    
    return RPP_SUCCESS;  
}

/************ gamma_correction ************/

template <typename T>
RppStatus gamma_correction_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                                      Rpp32f *batch_gamma, 
                                      RppiROI *roiPoints, Rpp32u nbatchSize,
                                      RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
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

            Rpp32f gamma = batch_gamma[batchCount];
            
            Rpp8u *gammaLUT = (Rpp8u *)calloc(256, sizeof(Rpp8u));
    
            for (int i = 0; i < 256; i++)
            {
                gammaLUT[i] = (T) RPPPIXELCHECK(pow((((Rpp32f) i) / 255.0), gamma) * 255.0);
            }
            
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
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                *dstPtrTemp = gammaLUT[*srcPtrTemp];

                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                            else
                            {
                                *dstPtrTemp = *srcPtrTemp;

                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
            free(gammaLUT);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
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

            Rpp32f gamma = batch_gamma[batchCount];

            Rpp8u *gammaLUT = (Rpp8u *)calloc(256, sizeof(Rpp8u));
    
            for (int i = 0; i < 256; i++)
            {
                gammaLUT[i] = (T) RPPPIXELCHECK(pow((((Rpp32f) i) / 255.0), gamma) * 255.0);
            }
            
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
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));

                            dstPtrTemp += channel;
                            srcPtrTemp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                *dstPtrTemp = gammaLUT[*srcPtrTemp];

                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }
            free(gammaLUT);
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus gamma_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f gamma,
                                RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageBuffer = channel * srcSize.height * srcSize.width;
    
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp8u *gammaLUT = (Rpp8u *)calloc(256, sizeof(Rpp8u));
    
    for (int i = 0; i < 256; i++)
    {
        gammaLUT[i] = (T) RPPPIXELCHECK(pow((((Rpp32f) i) / 255.0), gamma) * 255.0);
    }
    
    for (int i = 0; i < imageBuffer; i++)
    {
        *dstPtrTemp = gammaLUT[*srcPtrTemp];
        srcPtrTemp++;
        dstPtrTemp++;
    }

    free(gammaLUT);

    return RPP_SUCCESS;

}

/************ exposure ************/

template <typename T>
RppStatus exposure_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                              Rpp32f *batch_exposureFactor, 
                              RppiROI *roiPoints, Rpp32u nbatchSize,
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
            
            Rpp32f exposureFactor = batch_exposureFactor[batchCount];
            Rpp32f multiplyingFactor = pow(2, exposureFactor);
            
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
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                        Rpp32u alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128 pMul = _mm_set1_ps(multiplyingFactor);
                        __m128 p0, p1, p2, p3;
                        __m128i px0, px1, px2, px3;

                        int i = 0;
                        for (; i < alignedLength; i+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                            
                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
                            
                            px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, pMul));
                            px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, pMul));
                            px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, pMul));
                            px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, pMul));
                            
                            px0 = _mm_packus_epi32(px0, px1);
                            px1 = _mm_packus_epi32(px2, px3);
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
                            
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            
                            srcPtrTemp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; i < bufferLength; i++)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * multiplyingFactor);
                        }

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
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
            
            Rpp32f exposureFactor = batch_exposureFactor[batchCount];
            Rpp32f multiplyingFactor = pow(2, exposureFactor);
            
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

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;
                    Rpp32u alignedLength = bufferLength & ~15;

                    __m128i const zero = _mm_setzero_si128();
                    __m128 pMul = _mm_set1_ps(multiplyingFactor);
                    __m128 p0, p1, p2, p3;
                    __m128i px0, px1, px2, px3;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                        
                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
                        
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, pMul));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, pMul));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, pMul));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, pMul));
                        
                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
                        
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        
                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * multiplyingFactor);
                    }

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus exposure_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f exposureFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;

    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f multiplyingFactor = pow(2, exposureFactor);
    
    int bufferLength = channel * srcSize.height * srcSize.width;
    int alignedLength = bufferLength & ~15;

    __m128i const zero = _mm_setzero_si128();
    __m128 pMul = _mm_set1_ps(multiplyingFactor);
    __m128 p0, p1, p2, p3;
    __m128i px0, px1, px2, px3;

    int vectorLoopCount = 0;
    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
    {
        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
        
        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
        
        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, pMul));
        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, pMul));
        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, pMul));
        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, pMul));
        
        px0 = _mm_packus_epi32(px0, px1);
        px1 = _mm_packus_epi32(px2, px3);
        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
        
        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
        
        srcPtrTemp +=16;
        dstPtrTemp +=16;
    }
    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
    {
        *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * multiplyingFactor);
    }

    return RPP_SUCCESS;
}

/**************** blur ***************/

template <typename T>
RppStatus blur_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                                Rpp32u *batch_kernelSize,
                                RppiROI *roiPoints, Rpp32u nbatchSize,
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

            Rpp32u kernelSize = batch_kernelSize[batchCount];
            Rpp32f divisor = kernelSize * kernelSize;
            Rpp32u bound = (kernelSize - 1) / 2;

            Rpp32u firstRow = y1 + bound;
            Rpp32u firstColumn = x1 + bound;
            Rpp32u lastRow = y2 - bound;
            Rpp32u lastColumn = x2 - bound;

            Rpp32u roiWidthToCompute = lastColumn - firstColumn + 1;
            Rpp32u remainingElementsAfterROI = batch_srcSize[batchCount].width - roiWidthToCompute;
            Rpp32u incrementToNextKernel = (kernelSize * batch_srcSizeMax[batchCount].width) - 1;

            Rpp32u sums[25] = {0};
            Rpp32f pixel;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            T *srcPtrChannel, *dstPtrChannel;
            T *srcPtrTemp2, *srcPtrTemp3, *srcPtrTemp4;

            for(int c = 0; c < channel; c++)
            {
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);
                
                srcPtrTemp2 = srcPtrChannel + ((firstRow - bound) * batch_srcSizeMax[batchCount].width) + (firstColumn - bound);

                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (i < firstRow || i > lastRow)
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, firstColumn * sizeof(T));
                        srcPtrTemp += firstColumn;
                        dstPtrTemp += firstColumn;

                        srcPtrTemp3 = srcPtrTemp2;

                        pixel = 0;
                        for (int m = 0; m < kernelSize; m++)
                        {
                            sums[m] = 0;
                            for (int n = 0; n < kernelSize; n++)
                            {
                                sums[m] += *(srcPtrTemp3 + n);
                            }
                            srcPtrTemp3 += batch_srcSizeMax[batchCount].width;
                        }
                        for (int m = 0; m < kernelSize; m++)
                        {
                            pixel += sums[m];
                        }

                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel / divisor);
                        srcPtrTemp++;

                        srcPtrTemp3 = srcPtrTemp2;
                        srcPtrTemp4 = srcPtrTemp2 + kernelSize;

                        for (int j= 1; j < roiWidthToCompute; j++)
                        {
                            pixel = 0;
                            for (int m = 0; m < kernelSize; m++)
                            {
                                sums[m] = sums[m] - (Rpp32f) *srcPtrTemp3 + (Rpp32f) *srcPtrTemp4;
                                srcPtrTemp3 += batch_srcSizeMax[batchCount].width;
                                srcPtrTemp4 += batch_srcSizeMax[batchCount].width;
                            }
                            for (int m = 0; m < kernelSize; m++)
                            {
                                pixel += sums[m];
                            }

                            *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel / divisor);
                            srcPtrTemp++;
                            srcPtrTemp3 -= incrementToNextKernel;
                            srcPtrTemp4 -= incrementToNextKernel;
                        }
                        srcPtrTemp2 += batch_srcSizeMax[batchCount].width;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
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

            Rpp32u kernelSize = batch_kernelSize[batchCount];
            Rpp32f divisor = kernelSize * kernelSize;
            Rpp32u bound = (kernelSize - 1) / 2;

            Rpp32u firstRow = y1 + bound;
            Rpp32u firstColumn = x1 + bound;
            Rpp32u lastRow = y2 - bound;
            Rpp32u lastColumn = x2 - bound;

            Rpp32u roiWidthToCompute = lastColumn - firstColumn + 1;

            Rpp32u sumsR[25] = {0}, sumsG[25] = {0}, sumsB[25] = {0};
            Rpp32f pixelR, pixelG, pixelB;
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInKernelRow = channel * kernelSize;
            Rpp32u incrementToNextKernel = (kernelSize * elementsInRowMax) - channel;
            Rpp32u channeledBound = channel * bound;
            Rpp32u channeledFirstColumn = channel * firstColumn;

            T *srcPtrTemp2R, *srcPtrTemp2G, *srcPtrTemp2B;
            T *srcPtrTemp3R, *srcPtrTemp3G, *srcPtrTemp3B;
            T *srcPtrTemp4R, *srcPtrTemp4G, *srcPtrTemp4B;

            srcPtrTemp2R = srcPtrImage + ((firstRow - bound) * elementsInRowMax) + ((firstColumn - bound) * channel);
            srcPtrTemp2G = srcPtrTemp2R + 1;
            srcPtrTemp2B = srcPtrTemp2R + 2;
            
            // #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (i < firstRow || i > lastRow)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, channeledFirstColumn * sizeof(T));
                    dstPtrTemp += channeledFirstColumn;
                    srcPtrTemp += channeledFirstColumn;

                    srcPtrTemp3R = srcPtrTemp2R;
                    srcPtrTemp3G = srcPtrTemp2G;
                    srcPtrTemp3B = srcPtrTemp2B;
                    
                    pixelR = 0;
                    pixelG = 0;
                    pixelB = 0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        sumsR[m] = 0;
                        sumsG[m] = 0;
                        sumsB[m] = 0;
                        for (int n = 0; n < elementsInKernelRow; n += channel)
                        {
                            sumsR[m] += *(srcPtrTemp3R + n);
                            sumsG[m] += *(srcPtrTemp3G + n);
                            sumsB[m] += *(srcPtrTemp3B + n);
                        }
                        srcPtrTemp3R += elementsInRowMax;
                        srcPtrTemp3G += elementsInRowMax;
                        srcPtrTemp3B += elementsInRowMax;
                    }
                    for (int m = 0; m < kernelSize; m++)
                    {
                        pixelR += sumsR[m];
                        pixelG += sumsG[m];
                        pixelB += sumsB[m];
                    }

                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR / divisor);
                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG / divisor);
                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB / divisor);
                    srcPtrTemp += channel;

                    srcPtrTemp3R = srcPtrTemp2R;
                    srcPtrTemp3G = srcPtrTemp2G;
                    srcPtrTemp3B = srcPtrTemp2B;
                    srcPtrTemp4R = srcPtrTemp2R + elementsInKernelRow;
                    srcPtrTemp4G = srcPtrTemp2G + elementsInKernelRow;
                    srcPtrTemp4B = srcPtrTemp2B + elementsInKernelRow;

                    for (int j= 1; j < roiWidthToCompute; j++)
                    {
                        pixelR = 0;
                        pixelG = 0;
                        pixelB = 0;
                        for (int m = 0; m < kernelSize; m++)
                        {
                            sumsR[m] = sumsR[m] - (Rpp32f) *srcPtrTemp3R + (Rpp32f) *srcPtrTemp4R;
                            sumsG[m] = sumsG[m] - (Rpp32f) *srcPtrTemp3G + (Rpp32f) *srcPtrTemp4G;
                            sumsB[m] = sumsB[m] - (Rpp32f) *srcPtrTemp3B + (Rpp32f) *srcPtrTemp4B;
                            srcPtrTemp3R += elementsInRowMax;
                            srcPtrTemp3G += elementsInRowMax;
                            srcPtrTemp3B += elementsInRowMax;
                            srcPtrTemp4R += elementsInRowMax;
                            srcPtrTemp4G += elementsInRowMax;
                            srcPtrTemp4B += elementsInRowMax;
                        }
                        for (int m = 0; m < kernelSize; m++)
                        {
                            pixelR += sumsR[m];
                            pixelG += sumsG[m];
                            pixelB += sumsB[m];
                        }

                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR / divisor);
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG / divisor);
                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB / divisor);
                        srcPtrTemp += channel;
                        srcPtrTemp3R -= incrementToNextKernel;
                        srcPtrTemp3G -= incrementToNextKernel;
                        srcPtrTemp3B -= incrementToNextKernel;
                        srcPtrTemp4R -= incrementToNextKernel;
                        srcPtrTemp4G -= incrementToNextKernel;
                        srcPtrTemp4B -= incrementToNextKernel;
                    }

                    srcPtrTemp2R += elementsInRowMax;
                    srcPtrTemp2G += elementsInRowMax;
                    srcPtrTemp2B += elementsInRowMax;

                    memcpy(dstPtrTemp, srcPtrTemp, channeledFirstColumn * sizeof(T));
                    dstPtrTemp += channeledFirstColumn;
                    srcPtrTemp += channeledFirstColumn;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus blur_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f divisor = kernelSize * kernelSize;

    Rpp32u imageDim = srcSize.height * srcSize.width;
    Rpp32u bound = (kernelSize - 1) / 2;

    Rpp32u lastRow = srcSize.height - 1 - bound;
    Rpp32u lastColumn = srcSize.width - 1 - bound;

    Rpp32u widthToCompute = srcSize.width - (2 * bound);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        T *srcPtrTemp, *srcPtrTemp2, *srcPtrTemp3, *srcPtrTemp4, *dstPtrTemp;
        
        Rpp32u sums[25] = {0};
        Rpp32f pixel;

        Rpp32u incrementToNextKernel = (kernelSize * srcSize.width)  - 1;

        for (int c = 0; c < channel; c++)
        {
            T *srcPtrTemp = srcPtr + (c * imageDim);
            T *dstPtrTemp = dstPtr + (c * imageDim);

            srcPtrTemp2 = srcPtrTemp;

            for (int i = 0; i < srcSize.height; i++)
            {
                if (i < bound || i > lastRow)
                {
                    memcpy(dstPtrTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    srcPtrTemp += srcSize.width;
                    dstPtrTemp += srcSize.width;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, bound * sizeof(T));
                    dstPtrTemp += bound;
                    srcPtrTemp += bound;

                    srcPtrTemp3 = srcPtrTemp2;
                            
                    pixel = 0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        sums[m] = 0;
                        for (int n = 0; n < kernelSize; n++)
                        {
                            sums[m] += *(srcPtrTemp3 + n);
                        }
                        srcPtrTemp3 += srcSize.width;
                    }
                    for (int m = 0; m < kernelSize; m++)
                    {
                        pixel += sums[m];
                    }

                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel / divisor);
                    srcPtrTemp++;

                    srcPtrTemp3 = srcPtrTemp2;
                    srcPtrTemp4 = srcPtrTemp2 + kernelSize;

                    for (int j= 1; j < widthToCompute; j++)
                    {
                        pixel = 0;
                        for (int m = 0; m < kernelSize; m++)
                        {
                            sums[m] = sums[m] - (Rpp32f) *srcPtrTemp3 + (Rpp32f) *srcPtrTemp4;
                            srcPtrTemp3 += srcSize.width;
                            srcPtrTemp4 += srcSize.width;
                        }
                        for (int m = 0; m < kernelSize; m++)
                        {
                            pixel += sums[m];
                        }

                        *dstPtrTemp++ = (T) RPPPIXELCHECK(pixel / divisor);
                        srcPtrTemp++;
                        srcPtrTemp3 -= incrementToNextKernel;
                        srcPtrTemp4 -= incrementToNextKernel;
                    }
                    srcPtrTemp2 += srcSize.width;

                    memcpy(dstPtrTemp, srcPtrTemp, bound * sizeof(T));
                    dstPtrTemp += bound;
                    srcPtrTemp += bound;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        T *srcPtrTemp, *dstPtrTemp;
        T *srcPtrTemp2R, *srcPtrTemp2G, *srcPtrTemp2B;
        T *srcPtrTemp3R, *srcPtrTemp3G, *srcPtrTemp3B;
        T *srcPtrTemp4R, *srcPtrTemp4G, *srcPtrTemp4B;

        Rpp32u sumsR[25] = {0}, sumsG[25] = {0}, sumsB[25] = {0};
        Rpp32f pixelR, pixelG, pixelB;

        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u elementsInKernelRow = channel * kernelSize;
        Rpp32u incrementToNextKernel = (kernelSize * elementsInRow) - channel;
        Rpp32u channeledBound = channel * bound;

        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;
        srcPtrTemp2R = srcPtrTemp;
        srcPtrTemp2G = srcPtrTemp + 1;
        srcPtrTemp2B = srcPtrTemp + 2;

        for (int i = 0; i < srcSize.height; i++)
        {
            if (i < bound || i > lastRow)
            {
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                srcPtrTemp += elementsInRow;
                dstPtrTemp += elementsInRow;
            }
            else
            {
                memcpy(dstPtrTemp, srcPtrTemp, channeledBound * sizeof(T));
                dstPtrTemp += channeledBound;
                srcPtrTemp += channeledBound;

                srcPtrTemp3R = srcPtrTemp2R;
                srcPtrTemp3G = srcPtrTemp2G;
                srcPtrTemp3B = srcPtrTemp2B;
                
                pixelR = 0;
                pixelG = 0;
                pixelB = 0;
                for (int m = 0; m < kernelSize; m++)
                {
                    sumsR[m] = 0;
                    sumsG[m] = 0;
                    sumsB[m] = 0;
                    for (int n = 0; n < elementsInKernelRow; n += channel)
                    {
                        sumsR[m] += *(srcPtrTemp3R + n);
                        sumsG[m] += *(srcPtrTemp3G + n);
                        sumsB[m] += *(srcPtrTemp3B + n);
                    }
                    srcPtrTemp3R += elementsInRow;
                    srcPtrTemp3G += elementsInRow;
                    srcPtrTemp3B += elementsInRow;
                }
                for (int m = 0; m < kernelSize; m++)
                {
                    pixelR += sumsR[m];
                    pixelG += sumsG[m];
                    pixelB += sumsB[m];
                }

                *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR / divisor);
                *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG / divisor);
                *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB / divisor);
                srcPtrTemp += channel;

                srcPtrTemp3R = srcPtrTemp2R;
                srcPtrTemp3G = srcPtrTemp2G;
                srcPtrTemp3B = srcPtrTemp2B;
                srcPtrTemp4R = srcPtrTemp2R + elementsInKernelRow;
                srcPtrTemp4G = srcPtrTemp2G + elementsInKernelRow;
                srcPtrTemp4B = srcPtrTemp2B + elementsInKernelRow;

                for (int j= 1; j < widthToCompute; j++)
                {
                    pixelR = 0;
                    pixelG = 0;
                    pixelB = 0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        sumsR[m] = sumsR[m] - (Rpp32f) *srcPtrTemp3R + (Rpp32f) *srcPtrTemp4R;
                        sumsG[m] = sumsG[m] - (Rpp32f) *srcPtrTemp3G + (Rpp32f) *srcPtrTemp4G;
                        sumsB[m] = sumsB[m] - (Rpp32f) *srcPtrTemp3B + (Rpp32f) *srcPtrTemp4B;
                        srcPtrTemp3R += elementsInRow;
                        srcPtrTemp3G += elementsInRow;
                        srcPtrTemp3B += elementsInRow;
                        srcPtrTemp4R += elementsInRow;
                        srcPtrTemp4G += elementsInRow;
                        srcPtrTemp4B += elementsInRow;
                    }
                    for (int m = 0; m < kernelSize; m++)
                    {
                        pixelR += sumsR[m];
                        pixelG += sumsG[m];
                        pixelB += sumsB[m];
                    }

                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelR / divisor);
                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelG / divisor);
                    *dstPtrTemp++ = (T) RPPPIXELCHECK(pixelB / divisor);
                    srcPtrTemp += channel;
                    srcPtrTemp3R -= incrementToNextKernel;
                    srcPtrTemp3G -= incrementToNextKernel;
                    srcPtrTemp3B -= incrementToNextKernel;
                    srcPtrTemp4R -= incrementToNextKernel;
                    srcPtrTemp4G -= incrementToNextKernel;
                    srcPtrTemp4B -= incrementToNextKernel;
                }
                srcPtrTemp2R += elementsInRow;
                srcPtrTemp2G += elementsInRow;
                srcPtrTemp2B += elementsInRow;

                memcpy(dstPtrTemp, srcPtrTemp, channeledBound * sizeof(T));
                dstPtrTemp += channeledBound;
                srcPtrTemp += channeledBound;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** histogram_balance ***************/

template <typename T>
RppStatus histogram_balance_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                            Rpp32u nbatchSize,
                                            RppiChnFormat chnFormat, Rpp32u channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            Rpp32u bins = 256;
            Rpp8u bins8u = (Rpp8u) (((Rpp32u)(bins))- 1);

            Rpp32u *outputHistogramImage = (Rpp32u*) calloc(bins, sizeof(Rpp32u));
            T *lookUpTable = (T *) calloc (bins, sizeof(T));
            Rpp32f multiplier = 255.0 / ((Rpp32f)(channel * imageDim));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            if (bins == 0)
            {
                *outputHistogramImage = channel * imageDim;
            }
            else
            {
                Rpp8u rangeInBin = 256 / (bins8u + 1);

                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);

                    // #pragma omp parallel for
                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            #pragma omp critical
                            *(outputHistogramImage + (*srcPtrTemp / rangeInBin)) += 1;

                            srcPtrTemp++;
                        }
                    }
                }
            }

            Rpp32u sum = 0;
            Rpp32u *outputHistogramImageTemp;
            T *lookUpTableTemp;
            outputHistogramImageTemp = outputHistogramImage;
            lookUpTableTemp = lookUpTable;

            for (int i = 0; i < 256; i++)
            {
                sum += *outputHistogramImageTemp;
                *lookUpTableTemp = (T)round(((Rpp32f)sum) * multiplier);
                outputHistogramImageTemp++;
                lookUpTableTemp++;
            }

            Rpp32f x1 = 0;
            Rpp32f y1 = 0;
            Rpp32f x2 = 0;
            Rpp32f y2 = 0;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                //roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                //roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }
            
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    
                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            {
                                *dstPtrTemp = *(lookUpTable + *srcPtrTemp);
                            }
                            else
                            {
                                *dstPtrTemp = *srcPtrTemp;
                            }
                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }

            free(outputHistogramImage);
            free(lookUpTable);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u imageDim = batch_srcSize[batchCount].height * batch_srcSize[batchCount].width;

            Rpp32u bins = 256;
            Rpp8u bins8u = (Rpp8u) (((Rpp32u)(bins))- 1);

            Rpp32u *outputHistogramImage = (Rpp32u*) calloc(bins, sizeof(Rpp32u));
            T *lookUpTable = (T *) calloc (bins, sizeof(T));
            Rpp32f multiplier = 255.0 / ((Rpp32f)(channel * imageDim));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            if (bins == 0)
            {
                *outputHistogramImage = channel * imageDim;
            }
            else
            {
                Rpp8u rangeInBin = 256 / (bins8u + 1);

                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *(outputHistogramImage + (*srcPtrTemp / rangeInBin)) += 1;
                            
                            srcPtrTemp++;
                        }
                    }
                }
            }

            Rpp32u sum = 0;
            Rpp32u *outputHistogramImageTemp;
            T *lookUpTableTemp;
            outputHistogramImageTemp = outputHistogramImage;
            lookUpTableTemp = lookUpTable;

            for (int i = 0; i < 256; i++)
            {
                sum += *outputHistogramImageTemp;
                *lookUpTableTemp = (T)round(((Rpp32f)sum) * multiplier);
                outputHistogramImageTemp++;
                lookUpTableTemp++;
            }

            Rpp32f x1 = 0;
            Rpp32f y1 = 0;
            Rpp32f x2 = 0;
            Rpp32f y2 = 0;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                //roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                //roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }
            

            // #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                }
                else
                {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));
                            dstPtrTemp += channel;
                            srcPtrTemp += channel;
                        }
                        else
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                *dstPtrTemp = *(lookUpTable + *srcPtrTemp);

                                srcPtrTemp++;
                                dstPtrTemp++;
                            }
                        }
                    }
                }
            }

            free(outputHistogramImage);
            free(lookUpTable);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus histogram_balance_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat,Rpp32u channel)
{
    Rpp32u histogram[256];
    T lookUpTable[256];
    Rpp32u *histogramTemp;
    T *lookUpTableTemp;
    Rpp32f multiplier = 255.0 / ((Rpp32f)(channel * srcSize.height * srcSize.width));

    histogram_kernel_host(srcPtr, srcSize, histogram, 255, channel);

    Rpp32u sum = 0;
    histogramTemp = histogram;
    lookUpTableTemp = lookUpTable;
    
    for (int i = 0; i < 256; i++)
    {
        sum += *histogramTemp;
        *lookUpTableTemp = (T)round(((Rpp32f)sum) * multiplier);
        histogramTemp++;
        lookUpTableTemp++;
    }

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *(lookUpTable + *srcPtrTemp);
        srcPtrTemp++;
        dstPtrTemp++;
    }
    
    return RPP_SUCCESS;

}

/**************** random_crop_letterbox ***************/

template <typename T>
RppStatus random_crop_letterbox_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax, 
                                           Rpp32u *batch_x1, Rpp32u *batch_y1, Rpp32u *batch_x2, Rpp32u *batch_y2, RppiROI *roiPoints,
                                           Rpp32u nbatchSize,
                                           RppiChnFormat chnFormat, Rpp32u channel)
{
    #pragma omp parallel for
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32u x1 = batch_x1[batchCount];
        Rpp32u y1 = batch_y1[batchCount];
        Rpp32u x2 = batch_x2[batchCount];
        Rpp32u y2 = batch_y2[batchCount];

        // if ((RPPINRANGE(x1, 0, batch_srcSize[batchCount].width - 1) == 0) 
        // || (RPPINRANGE(x2, 0, batch_srcSize[batchCount].width - 1) == 0) 
        // || (RPPINRANGE(y1, 0, batch_srcSize[batchCount].height - 1) == 0) 
        // || (RPPINRANGE(y2, 0, batch_srcSize[batchCount].height - 1) == 0))
        // {
        //     return RPP_ERROR;
        // }

        Rpp32u srcLoc = 0, dstLoc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
        compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);

        T *srcPtrImage = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        T *dstPtrImage = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

        compute_unpadded_from_padded_host(srcPtr + srcLoc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtrImage, 
                                            chnFormat, channel);
        
        Rpp32u borderWidth = (5 * RPPMIN2(batch_dstSize[batchCount].height, batch_dstSize[batchCount].width) / 100);

        RppiSize srcSizeSubImage;
        T* srcPtrSubImage;
        compute_subimage_location_host(srcPtrImage, &srcPtrSubImage, batch_srcSize[batchCount], &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

        RppiSize srcSizeSubImagePadded;
        srcSizeSubImagePadded.height = srcSizeSubImage.height + (2 * borderWidth);
        srcSizeSubImagePadded.width = srcSizeSubImage.width + (2 * borderWidth);

        T *srcPtrImageCrop = (T *)calloc(channel * srcSizeSubImage.height * srcSizeSubImage.width, sizeof(T));
        generate_crop_host(srcPtrImage, batch_srcSize[batchCount], srcPtrSubImage, srcSizeSubImage, srcPtrImageCrop, chnFormat, channel);

        T *srcPtrImageCropPadded = (T *)calloc(channel * srcSizeSubImagePadded.height * srcSizeSubImagePadded.width, sizeof(T));
        generate_evenly_padded_image_host(srcPtrImageCrop, srcSizeSubImage, srcPtrImageCropPadded, srcSizeSubImagePadded, chnFormat, channel);

        resize_kernel_host(srcPtrImageCropPadded, srcSizeSubImagePadded, dstPtrImage, batch_dstSize[batchCount], chnFormat, channel);

        compute_padded_from_unpadded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtr + dstLoc, 
                                          chnFormat, channel);
        
        free(srcPtrImage);
        free(dstPtrImage);
        free(srcPtrImageCrop);
        free(srcPtrImageCropPadded);
    }

    return RPP_SUCCESS;       
}

template <typename T>
RppStatus random_crop_letterbox_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                                     Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if ((RPPINRANGE(x1, 0, srcSize.width - 1) == 0) 
        || (RPPINRANGE(x2, 0, srcSize.width - 1) == 0) 
        || (RPPINRANGE(y1, 0, srcSize.height - 1) == 0) 
        || (RPPINRANGE(y2, 0, srcSize.height - 1) == 0))
    {
        return RPP_ERROR;
    }

    Rpp32u borderWidth = (5 * RPPMIN2(dstSize.height, dstSize.width) / 100);

    RppiSize srcSizeSubImage;
    T* srcPtrSubImage;
    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    RppiSize srcSizeSubImagePadded;
    srcSizeSubImagePadded.height = srcSizeSubImage.height + (2 * borderWidth);
    srcSizeSubImagePadded.width = srcSizeSubImage.width + (2 * borderWidth);

    T *srcPtrCrop = (T *)calloc(channel * srcSizeSubImage.height * srcSizeSubImage.width, sizeof(T));
    generate_crop_host(srcPtr, srcSize, srcPtrSubImage, srcSizeSubImage, srcPtrCrop, chnFormat, channel);

    T *srcPtrCropPadded = (T *)calloc(channel * srcSizeSubImagePadded.height * srcSizeSubImagePadded.width, sizeof(T));
    generate_evenly_padded_image_host(srcPtrCrop, srcSizeSubImage, srcPtrCropPadded, srcSizeSubImagePadded, chnFormat, channel);

    resize_kernel_host(srcPtrCropPadded, srcSizeSubImagePadded, dstPtr, dstSize, chnFormat, channel);

    free(srcPtrCrop);
    free(srcPtrCropPadded);

    return RPP_SUCCESS;
}



/**************** Pixelate ***************/

template <typename T>
RppStatus pixelate_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;
            
            if( x2 != batch_srcSize[batchCount].width && y2 != batch_srcSize[batchCount].height)
            {
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannel, *dstPtrChannel;
                    srcPtrChannel = srcPtrImage + (c * imageDimMax);
                    dstPtrChannel = dstPtrImage + (c * imageDimMax);

                    // #pragma omp parallel for
                    for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                    {
                        T *srcPtrTemp, *dstPtrTemp;
                        srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                }
            }
            for(int c = 0; c < channel; c++)
            {
                
                for(int i = 0; i < batch_srcSize[batchCount].height; )
                {
                    bool roi = false;
                    for(int j = 0; j < batch_srcSize[batchCount].width;)
                    {
                        if((i >= y1) && (i <= y2 ) && ( j >= x1) && (j <= x2))
                        {
                            if( i + 7 < batch_srcSize[batchCount].height && j + 7 < batch_srcSize[batchCount].width)
                            {
                                int sum = 0;
                                roi = true;
                                
                                // #pragma omp parallel for
                                for(int bi = 0 ; bi < 7 ; bi++)
                                {
                                    for(int bj = 0 ; bj < 7 ; bj++)
                                    {
                                        if(i + bi < batch_srcSize[batchCount].height && j + bj < batch_srcSize[batchCount].width && i + bi >= 0 && j + bj >= 0)
                                        {    
                                            sum += *(srcPtrImage + ((i + bi) * batch_srcSizeMax[batchCount].width + (j + bj) + c * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width));
                                        }
                                    }
                                }
                                sum /= 49;

                                for(int bi = 0 ; bi < 7 ; bi++)
                                {
                                    for(int bj = 0 ; bj < 7 ; bj++)
                                    {
                                        if(i + bi < batch_srcSize[batchCount].height && j + bj < batch_srcSize[batchCount].width)
                                        {    
                                            *(dstPtrImage + ((i + bi) * batch_srcSizeMax[batchCount].width + (j + bj) + c * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)) = RPPPIXELCHECK(sum);
                                        }
                                    }
                                }
                            }
                        }   
                        if (roi)
                            j+=7;
                        else
                            j++;
                    }
                    if(roi)
                    {
                        roi = false;
                        i+= 7;
                    }
                    else
                        i++;
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
            
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            
            int te =0;
            if( x2 != batch_srcSize[batchCount].width && y2 != batch_srcSize[batchCount].height)
            {
                // #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
            }
            for(int i = 0; i < batch_srcSize[batchCount].height; )
            {
                bool roi = false;
                for(int j = 0; j < batch_srcSize[batchCount].width;)
                {
                    if((i >= y1) && (i <= y2 ) && ( j >= x1) && (j <= x2))
                    {
                        if( i + 7 < batch_srcSize[batchCount].height && j + 7 < batch_srcSize[batchCount].width)
                        {
                            int sum = 0;
                            for(int c = 0; c < channel; c++)
                            {
                                roi = true;
                                
                                // #pragma omp parallel for
                                for(int bi = 0 ; bi < 7 ; bi++)
                                {
                                    for(int bj = 0 ; bj < 7 ; bj++)
                                    {
                                        if(i + bi < batch_srcSize[batchCount].height && j + bj < batch_srcSize[batchCount].width && i + bi >= 0 && j + bj >= 0)
                                        {    
                                            sum += *(srcPtrImage + ((i + bi) * batch_srcSizeMax[batchCount].width * channel + (j + bj) * channel + c));
                                        }
                                    }
                                }
                                sum /= 49;

                                for(int bi = 0 ; bi < 7 ; bi++)
                                {
                                    for(int bj = 0 ; bj < 7 ; bj++)
                                    {
                                        if(i + bi < batch_srcSize[batchCount].height && j + bj < batch_srcSize[batchCount].width)
                                        {    
                                            *(dstPtrImage + ((i + bi) * batch_srcSizeMax[batchCount].width * channel + (j + bj) * channel + c)) = RPPPIXELCHECK(sum);
                                        }
                                    }
                                }
                            }
                        }
                    }   
                    if (roi)
                        j+=7;
                    else
                        j++;
                }
                if(roi)
                {
                    roi = false;
                    i+= 7;
                }
                else
                    i++;
            }
        }
    }
    
    return RPP_SUCCESS;
}



template <typename T>
RppStatus pixelate_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *dstTemp,*srcTemp;
    dstTemp = dstPtr;
    srcTemp = srcPtr;
    int sum = 0;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        for(int y = 0 ; y < srcSize.height ;)
        {
            for(int x = 0 ; x < srcSize.width ;)
            {
                for(int c = 0 ; c < channel ; c++)
                {    
                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width && y + i >= 0 && x + j >= 0)
                            {    
                                sum += *(srcPtr + ((y + i) * srcSize.width * channel + (x + j) * channel + c));
                            }
                        }
                    }
                    sum /= 49;

                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width)
                            {    
                                *(dstTemp + ((y + i) * srcSize.width * channel + (x + j) * channel + c)) = RPPPIXELCHECK(sum);
                            }
                        }
                    }
                    sum = 0;
                }
                x +=7;
            }
            y += 7;
        }
    }    
    else
    {
        for(int c = 0 ; c < channel ; c++)
        {
            for(int y = 0 ; y < srcSize.height ;)
            {
                for(int x = 0 ; x < srcSize.width ;)    
                {    
                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width && y + i >= 0 && x + j >= 0)
                            {    
                                sum += *(srcPtr + (y + i) * srcSize.width + (x + j) + c * srcSize.height * srcSize.width);
                            }
                        }
                    }
                    sum /= 49;

                    for(int i = 0 ; i < 7 ; i++)
                    {
                        for(int j = 0 ; j < 7 ; j++)
                        {
                            if(y + i < srcSize.height && x + j < srcSize.width)
                            {    
                                *(dstTemp + (y + i) * srcSize.width + (x + j) + c * srcSize.height * srcSize.width) = RPPPIXELCHECK(sum);
                            }
                        }
                    }
                    sum = 0;
                    x +=7;
                }
                y += 7;
            }
        }
    }
    return RPP_SUCCESS;
}

/**************** Fog ***************/

template <typename T>
RppStatus fog_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                              Rpp32f *batch_fogValue, 
                              Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f fogValue = batch_fogValue[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;
            // if( x2 != batch_srcSize[batchCount].width && y2 != batch_srcSize[batchCount].height)
            // {
            //     for(int c = 0; c < channel; c++)
            //     {
            //         T *srcPtrChannel, *dstPtrChannel;
            //         srcPtrChannel = srcPtrImage + (c * imageDimMax);
            //         dstPtrChannel = dstPtrImage + (c * imageDimMax);

            //         //#pragma omp parallel for simd
                // #pragma omp parallel for
            //         for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            //         {
            //             T *srcPtrTemp, *dstPtrTemp;
            //             srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
            //             dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
            //             memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
            //             dstPtrTemp += batch_srcSizeMax[batchCount].width;
            //             srcPtrTemp += batch_srcSizeMax[batchCount].width;
            //         }
            //     }
            // }
            #pragma omp parallel for  
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    // if((i >= y1) && (i <= y2 ) && ( j >= x1) && (j <= x2)){
                    if(fogValue >= 0)
                    {
                        if (channel == 3){
                            Rpp8u dstPtrTemp1, dstPtrTemp2, dstPtrTemp3;
                            dstPtrTemp1 = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j);
                            dstPtrTemp2 = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j + batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height);
                            dstPtrTemp3 = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j + 2 * batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height);
                            Rpp32f check = (dstPtrTemp3 + dstPtrTemp1 + dstPtrTemp2) / 3;
                            *(dstPtrImage  + i * batch_srcSizeMax[batchCount].width + j) = fogGenerator(dstPtrTemp1, fogValue, 1, check);
                            *(dstPtrImage + i * batch_srcSizeMax[batchCount].width + j + batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height) = fogGenerator(dstPtrTemp2, fogValue, 2, check);
                            *(dstPtrImage + i * batch_srcSizeMax[batchCount].width + j + 2 * batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height) = fogGenerator(dstPtrTemp3, fogValue, 3, check);
                        }
                        if(channel == 1){
                            Rpp32f check = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j);
                            *(dstPtrImage  + i * batch_srcSizeMax[batchCount].width + j) = fogGenerator(check, fogValue, 1, check);
                        }
                    } else{
                        if (channel == 3){
                            *(dstPtrImage  + i * batch_srcSizeMax[batchCount].width + j) = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j);
                            *(dstPtrImage + i * batch_srcSizeMax[batchCount].width + j + batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height) = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j + batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height);
                            *(dstPtrImage + i * batch_srcSizeMax[batchCount].width + j + 2 * batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height) = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j + 2 * batch_srcSizeMax[batchCount].width * batch_srcSizeMax[batchCount].height);
                        }
                        if(channel == 1){
                            *(dstPtrImage  + i * batch_srcSizeMax[batchCount].width + j) = *(srcPtrImage + i * batch_srcSizeMax[batchCount].width + j);
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

            Rpp32f fogValue = batch_fogValue[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            T *srcPtrTemp1, *dstPtrTemp1;
            srcPtrTemp1 = srcPtrImage;
            dstPtrTemp1 = dstPtrImage;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                // if (!((y1 <= i) && (i <= y2)))
                // {
                //     memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                //     dstPtrTemp += elementsInRowMax;
                //     srcPtrTemp += elementsInRowMax;
                // }
                // else
                // {
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        // if (!((x1 <= j) && (j <= x2 )))
                        // {
                        //     memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));

                        //     dstPtrTemp += channel;
                        //     srcPtrTemp += channel;
                        // }
                        // else
                        // {
                            if(fogValue <= 0)
                            {
                                for(int i = 0;i < channel;i++)
                                {
                                    *dstPtrTemp = *srcPtrTemp;
                                    dstPtrTemp++;
                                    srcPtrTemp++;
                                }
                            }
                            if(fogValue != 0){
                                Rpp8u dstPtrTemp1, dstPtrTemp2, dstPtrTemp3;
                                dstPtrTemp1 = *srcPtrTemp++;
                                dstPtrTemp2 = *srcPtrTemp++;
                                dstPtrTemp3 = *srcPtrTemp++;
                                Rpp32f check = (dstPtrTemp3 + dstPtrTemp1 + dstPtrTemp2) / 3;
                                *dstPtrTemp = fogGenerator(dstPtrTemp1, fogValue, 1, check);
                                *(dstPtrTemp+1) = fogGenerator(dstPtrTemp2, fogValue, 2, check);
                                *(dstPtrTemp+2) = fogGenerator(dstPtrTemp3, fogValue, 3, check);
                                dstPtrTemp += channel;
                            }
                        // }
                    }
                // }
            }
        }
    }
    
    return RPP_SUCCESS;
}


template <typename T>
RppStatus fog_host(  T* temp,RppiSize srcSize,T* srcPtr, 
                    Rpp32f fogValue,
                    RppiChnFormat chnFormat,   unsigned int channel)
{
    // if(fogValue <= 0)
    // {
        Rpp8u *srcPtr1;
        srcPtr1 = srcPtr;
        for(int i = 0;i < srcSize.height * srcSize.width * channel;i++)
        {
            *srcPtr1 = *temp;
            srcPtr1++;
            temp++;
        }
    // }
    if(fogValue != 0)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            Rpp8u *srcPtr1, *srcPtr2;
            if(channel > 1)
            {
                srcPtr1 = srcPtr + (srcSize.width * srcSize.height);
                srcPtr2 = srcPtr + (srcSize.width * srcSize.height * 2);
            }
            for (int i = 0; i < (srcSize.width * srcSize.height); i++)
            {
                Rpp32f check= *srcPtr;
                if(channel > 1) 
                    check = (check + *srcPtr1 + *srcPtr2) / 3;
                *srcPtr = fogGenerator(*srcPtr, fogValue, 1, check);
                srcPtr++;
                if(channel > 1)
                {
                    *srcPtr1 = fogGenerator(*srcPtr1, fogValue, 2, check);
                    *srcPtr2 = fogGenerator(*srcPtr2, fogValue, 3, check);
                    srcPtr1++;
                    srcPtr2++;
                }
            }
        }
        else
        {
            Rpp8u *srcPtr1, *srcPtr2;
            srcPtr1 = srcPtr + 1;
            srcPtr2 = srcPtr + 2;
            for (int i = 0; i < (srcSize.width * srcSize.height * channel); i += 3)
            {
                Rpp32f check = (*srcPtr + *srcPtr1 + *srcPtr2) / 3;
                *srcPtr = fogGenerator(*srcPtr, fogValue, 1, check);
                *srcPtr1 = fogGenerator(*srcPtr1, fogValue, 2, check);
                *srcPtr2 = fogGenerator(*srcPtr2, fogValue, 3, check);
                srcPtr += 3;
                srcPtr1 += 3;
                srcPtr2 += 3;
            }
        }
    }
    return RPP_SUCCESS;

}

/**************** Noise ***************/
template <typename T>
RppStatus noise_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                              Rpp32f *batch_noiseProbability, 
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel)
{
    srand(time(0)); 
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
            if (x2 == 0){
                 x2 = batch_srcSize[batchCount].width;
            }
            if (y2 == 0){
                y2 = batch_srcSize[batchCount].height;
            }
            Rpp32f noiseProbability = batch_noiseProbability[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;
            Rpp32u noisePixel;
            // if(noiseProbability == 0)
            //     noiseProbability = 0.002;
            if(noiseProbability != 0){
                noisePixel = (Rpp32u)(noiseProbability * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height );
            }
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSizeMax[batchCount].width * channel * sizeof(T));
            }

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;
            if(noiseProbability != 0){
                //#pragma omp parallel for simd
                #pragma omp parallel for
                for(int i = 0 ; i < noisePixel ; i++)
                {
                    // Rpp32u row = rand() % batch_srcSize[batchCount].height;
                    // Rpp32u column = rand() % batch_srcSize[batchCount].width;
                    Rpp32u random = rand();
                    Rpp32u row = random % (y2 + 1 - y1) + y1;
                    Rpp32u column = random % (x2 + 1 - x1)+ x1;
                    Rpp8u newPixel = random%2 ? 0 : 255;
                    for(int k = 0;k < channel;k++)
                    {
                        dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] = newPixel;
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
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;
            Rpp32f noiseProbability = batch_noiseProbability[batchCount];
            // noiseProbability = 0;
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            Rpp32u noisePixel;
            // if(noiseProbability == 0)
            //     noiseProbability = 0.002;
            if(noiseProbability != 0){
                noisePixel = (Rpp32u)(noiseProbability * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height );
            }
            //#pragma omp parallel for simd
            #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
            }
            // std::cerr<<"\n pixeldistance :: "<<pixelDistance<<" noisePixel::"<<noisePixel;
            if(noiseProbability != 0)
            { 
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage;
                dstPtrTemp = dstPtrImage;
                //#pragma omp parallel for simd
                #pragma omp parallel for
                for(int i = 0 ; i < noisePixel ; i++)
                {
                    // Rpp32u row = rand() % batch_srcSize[batchCount].height;
                    // Rpp32u column = rand() % batch_srcSize[batchCount].width;
                    Rpp32u random = rand();
                    Rpp32u row =  (random % ((y2) - (y1) + 1)) + y1;
                    Rpp32u random1 = rand();
                    Rpp32u column = (random1 % ((x2) - (x1)+ 1))+ x1;
                    Rpp32u random2 = rand();
                    Rpp8u newPixel = random % 2 ? 0 : 255;

                    for(int k = 0;k < channel;k++)
                    {
                        dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] = newPixel;
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}



template <typename T>
RppStatus noise_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                        Rpp32f noiseProbability, 
                        RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp8u *cpdst,*cpsrc;
    cpdst = dstPtr;
    cpsrc = srcPtr;
    for (int i = 0; i < ( channel * srcSize.width * srcSize.height ); i++ )
    {
        *cpdst = *cpsrc;
        cpdst++;
        cpsrc++;
    }
    if(noiseProbability != 0)
    {           
        srand(time(0)); 
        Rpp32u noisePixel = (Rpp32u)(noiseProbability * srcSize.width * srcSize.height );
        Rpp32u pixelDistance = (srcSize.width * srcSize.height) / noisePixel;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            for(int i = 0 ; i < srcSize.width * srcSize.height * channel ; i += channel*pixelDistance)
            {
                Rpp32u initialPixel = rand() % pixelDistance;
                dstPtr += initialPixel*channel;
                Rpp8u newPixel = rand()%2 ? 0 : 255;
                for(int j = 0 ; j < channel ; j++)
                {
                    *dstPtr = newPixel;
                    dstPtr++;
                }
                dstPtr += ((pixelDistance - initialPixel - 1) * channel);
            }
        }
        else if(chnFormat == RPPI_CHN_PLANAR)
        {
            if(channel == 3)
            {
                Rpp8u *dstPtrTemp1,*dstPtrTemp2;
                dstPtrTemp1 = dstPtr + (srcSize.height * srcSize.width);
                dstPtrTemp2 = dstPtr + (2 * srcSize.height * srcSize.width);   
                for(int i = 0 ; i < srcSize.width * srcSize.height * channel ; i += pixelDistance)
                {
                    Rpp32u initialPixel = rand() % pixelDistance;
                    dstPtr += initialPixel;
                    Rpp8u newPixel = (rand() % 2) ? 255 : 1;
                    *dstPtr = newPixel;
                    dstPtr += ((pixelDistance - initialPixel - 1));

                    dstPtrTemp1 += initialPixel;
                    *dstPtrTemp1 = newPixel;
                    dstPtrTemp1 += ((pixelDistance - initialPixel - 1));

                    dstPtrTemp2 += initialPixel;
                    *dstPtrTemp2 = newPixel;
                    dstPtrTemp2 += ((pixelDistance - initialPixel - 1));
                    
                }
            }
            else
            {
                for(int i = 0 ; i < srcSize.width * srcSize.height ; i += pixelDistance)
                {
                    Rpp32u initialPixel = rand() % pixelDistance;
                    dstPtr += initialPixel;
                    Rpp8u newPixel = rand()%2 ? 255 : 1;
                    *dstPtr = newPixel;
                    dstPtr += ((pixelDistance - initialPixel - 1));
                }   
            }
            
        }
    }
    return RPP_SUCCESS;
}

/**************** Snow ***************/

template <typename T>
RppStatus snow_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                              Rpp32f *batch_strength, 
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;

            Rpp32f strength = batch_strength[batchCount];

            strength = strength/100;
            int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

            Rpp32u snowDrops = (Rpp32u)(strength * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height * channel );

            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSizeMax[batchCount].width * channel * sizeof(T));
                // dstPtrTemp += batch_srcSizeMax[batchCount].width;
                // srcPtrTemp += batch_srcSizeMax[batchCount].width;
            }

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0 ; i < snowDrops ; i++)
            {
                Rpp32u row = rand() % batch_srcSize[batchCount].height;
                Rpp32u column = rand() % batch_srcSize[batchCount].width;
                Rpp32f pixel;
                for(int k = 0;k < channel;k++)
                {
                    dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] = RPPPIXELCHECK(dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] + snow_mat[0][0]) ;
                }
                for(int j = 0;j < 5;j++)
                {
                    if(row + 5 < batch_srcSize[batchCount].height && row + 5 > 0 )
                    for(int k = 0;k < channel;k++)
                    {
                        for(int m = 0;m < 5;m++)
                        {
                            if (column + 5 < batch_srcSizeMax[batchCount].width && column + 5 > 0)
                            {
                                dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (batch_srcSizeMax[batchCount].width * j) + m] = RPPPIXELCHECK( dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (batch_srcSizeMax[batchCount].width * j) + m] + snow_mat[j][m]) ;
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
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;

            Rpp32f strength = batch_strength[batchCount];

            strength = strength/100;
            int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

            Rpp32u snowDrops = (Rpp32u)(strength * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height * channel );
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                // dstPtrTemp += elementsInRowMax;
                // srcPtrTemp += elementsInRowMax;
            }
            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0 ; i < snowDrops ; i++)
            {
                Rpp32u row = rand() % batch_srcSize[batchCount].height;
                Rpp32u column = rand() % batch_srcSize[batchCount].width;
                Rpp32f pixel;
                for(int k = 0;k < channel;k++)
                {
                    dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] = RPPPIXELCHECK(dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] + snow_mat[0][0]) ;
                }
                for(int j = 0;j < 5;j++)
                {
                    if(row + 5 < batch_srcSize[batchCount].height && row + 5 > 0 )
                    for(int k = 0;k < channel;k++)
                    {
                        for(int m = 0;m < 5;m++)
                        {
                            if (column + 5 < batch_srcSize[batchCount].width && column + 5 > 0)
                            {
                                dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k + (channel * batch_srcSizeMax[batchCount].width * j) + (channel * m)] = RPPPIXELCHECK( dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k + (channel * batch_srcSizeMax[batchCount].width * j) + (channel * m)] + snow_mat[j][m]);
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
RppStatus snow_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f strength,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    strength = strength/100;
    int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

    Rpp32u snowDrops = (Rpp32u)(strength * srcSize.width * srcSize.height * channel );
    
    T *dstptrtemp;
    dstptrtemp=dstPtr;
    for(int k=0;k<srcSize.height*srcSize.width*channel;k++)
    {
        *dstptrtemp = 0;
        dstptrtemp++;
    }
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < snowDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k = 0;k < channel;k++)
            {
                dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width)] = snow_mat[0][0] ;
            }
            for(int j = 0;j < 5;j++)
            {
                if(row + 5 < srcSize.height && row + 5 > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < 5;m++)
                    {
                        if (column + 5 < srcSize.width && column + 5 > 0)
                        {
                            dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width) + (srcSize.width * j) + m] = snow_mat[j][m] ;
                        }
                    }
                }            
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0 ; i < snowDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k = 0;k < channel;k++)
            {
                dstPtr[(channel * row * srcSize.width) + (column * channel) + k] = snow_mat[0][0] ;
            }
            for(int j = 0;j < 5;j++)
            {
                if(row + 5 < srcSize.height && row + 5 > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < 5;m++)
                    {
                        if (column + 5 < srcSize.width && column + 5 > 0)
                        {
                            dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j) + (channel * m)] = snow_mat[j][m];
                        }
                    } 
                }            
            }
        }
    }

    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32u pixel = ((Rpp32u) srcPtr[i]) + (Rpp32u)dstPtr[i];
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }
    return RPP_SUCCESS;
}


/**************** Rain ***************/
template <typename T>
RppStatus rain_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                              Rpp32f *batch_rainPercentage, Rpp32u *batch_rainWidth, Rpp32u *batch_rainHeight, Rpp32f *batch_transparency, 
                              Rpp32u nbatchSize, RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f rainPercentage = batch_rainPercentage[batchCount];
            Rpp32u rainWidth = batch_rainWidth[batchCount];
            Rpp32u rainHeight = batch_rainHeight[batchCount];
            Rpp32f transparency = batch_transparency[batchCount];
            rainPercentage = rainPercentage / 250;
            transparency /= 5;

            Rpp32u rainDrops = (Rpp32u)(rainPercentage * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height * channel );

            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSizeMax[batchCount].width * channel * sizeof(T));
                // dstPtrTemp += batch_srcSizeMax[batchCount].width;
                // srcPtrTemp += batch_srcSizeMax[batchCount].width;
            }

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0 ; i < rainDrops ; i++)
            {
                Rpp32u row = rand() % batch_srcSize[batchCount].height;
                Rpp32u column = rand() % batch_srcSize[batchCount].width;
                Rpp32f pixel;
                for(int k = 0;k < channel;k++)
                {
                    pixel = (k == 0) ? 196 : (k == 1) ? 226 : 255  * transparency;
                    dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] = RPPPIXELCHECK(dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] + pixel);
                }
                for(int j = 0;j < rainHeight;j++)
                {
                    if(row + rainHeight < batch_srcSize[batchCount].height && row + rainHeight > 0 )
                    for(int k = 0;k < channel;k++)
                    {
                        for(int m = 0;m < rainWidth;m++)
                        {
                            if (column + rainWidth < batch_srcSizeMax[batchCount].width && column + rainWidth > 0)
                            {
                                pixel = (k == 0) ? 196 * transparency : (k == 1) ? 226* transparency : 255 * transparency;
                                dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (batch_srcSizeMax[batchCount].width * j) + m] =  RPPPIXELCHECK(dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (batch_srcSizeMax[batchCount].width * j) + m] + pixel);
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
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32f rainPercentage = batch_rainPercentage[batchCount];
            Rpp32u rainWidth = batch_rainWidth[batchCount];
            Rpp32u rainHeight = batch_rainHeight[batchCount];
            Rpp32f transparency = batch_transparency[batchCount];

            rainPercentage = rainPercentage / 250;
            transparency /= 5;

            Rpp32u rainDrops = (Rpp32u)(rainPercentage * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height * channel );
    
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
                // dstPtrTemp += elementsInRowMax;
                // srcPtrTemp += elementsInRowMax;
            }
            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0 ; i < rainDrops ; i++)
            {
                Rpp32u row = rand() % batch_srcSize[batchCount].height;
                Rpp32u column = rand() % batch_srcSize[batchCount].width;
                Rpp32f pixel;
                for(int k = 0;k < channel;k++)
                {
                    pixel = (k == 0) ? 196 : (k == 1) ? 226 : 255 * transparency;
                    dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] = RPPPIXELCHECK(dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] + pixel);
                }
                for(int j = 0;j < rainHeight;j++)
                {
                    if(row + rainHeight < batch_srcSize[batchCount].height && row + rainHeight > 0 )
                    for(int k = 0;k < channel;k++)
                    {
                        for(int m = 0;m < rainWidth;m++)
                        {
                            if (column + rainWidth < batch_srcSize[batchCount].width && column + rainWidth > 0)
                            {

                                pixel = (k == 0) ? 196 : (k == 1) ? 226 : 255 * transparency;
                                dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k + (channel * batch_srcSizeMax[batchCount].width * j) + (channel * m)] = RPPPIXELCHECK(dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k + (channel * batch_srcSizeMax[batchCount].width * j) + (channel * m)] + pixel);
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
RppStatus rain_host(T* srcPtr, RppiSize srcSize,T* dstPtr,
                    Rpp32f rainPercentage, Rpp32f rainWidth, Rpp32f rainHeight, Rpp32f transparency,
                    RppiChnFormat chnFormat,   unsigned int channel)
{ 
    rainPercentage = rainPercentage / 250;
    transparency /= 5;

    Rpp32u rainDrops = (Rpp32u)(rainPercentage * srcSize.width * srcSize.height * channel );
    
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < rainDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k = 0;k < channel;k++)
            {
                //pixel=(Rpp32f)dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width)] + 5;
                dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width)] = (k == 0) ? 196 : (k == 1) ? 226 : 255 ;
            }
            for(int j = 0;j < rainHeight;j++)
            {
                if(row + rainHeight < srcSize.height && row + rainHeight > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < rainWidth;m++)
                    {
                        if (column + rainWidth < srcSize.width && column + rainWidth > 0)
                        {
                            //pixel=(Rpp32f)dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width) + (srcSize.width*j)+m]+5;
                            dstPtr[(row * srcSize.width) + (column) + (k * srcSize.height * srcSize.width) + (srcSize.width * j) + m] = (k == 0) ? 196 : (k == 1) ? 226 : 255 ;
                        }
                    }
                }            
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0 ; i < rainDrops ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k = 0;k < channel;k++)
            {
                //pixel=(Rpp32f)dstPtr[(channel * row * srcSize.width) + (column * channel) + k] + 5;
                dstPtr[(channel * row * srcSize.width) + (column * channel) + k] = (k == 0) ? 196 : (k == 1) ? 226 : 255 ;
            }
            for(int j = 0;j < rainHeight;j++)
            {
                if(row + rainHeight < srcSize.height && row + rainHeight > 0 )
                for(int k = 0;k < channel;k++)
                {
                    for(int m = 0;m < rainWidth;m++)
                    {
                        if (column + rainWidth < srcSize.width && column + rainWidth > 0)
                        {
                            //pixel=(Rpp32f)dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j)+(channel*m)]+5;
                            dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j) + (channel * m)] = (k == 0) ? 196 : (k == 1) ? 226 : 255 ;
                        }
                    } 
                }            
            }
        }
    }

    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) + transparency * dstPtr[i];
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }
    return RPP_SUCCESS;
}


/**************** Random Shadow ***************/

template <typename T>
RppStatus random_shadow_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                             Rpp32u *batch_x1, Rpp32u *batch_y1, Rpp32u *batch_x2, Rpp32u *batch_y2, 
                             Rpp32u *batch_numberOfShadows, Rpp32u *batch_maxSizeX, Rpp32u *batch_maxSizeY,
                             Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u numberOfShadows = batch_numberOfShadows[batchCount];
            Rpp32u maxSizeX = batch_maxSizeX[batchCount];
            Rpp32u maxSizeY = batch_maxSizeY[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSizeMax[batchCount].width * channel * sizeof(T));
            }

            srand (time(NULL));
            RppiSize srcSizeSubImage,shadowSize;
            T *srcPtrSubImage, *dstPtrSubImage;
            srcSizeSubImage.height = RPPABS(y2 - y1) + 1;
            srcSizeSubImage.width = RPPABS(x2 - x1) + 1;
            srcPtrSubImage = srcPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + (x1);
            dstPtrSubImage = dstPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + (x1);

            // if (maxSizeX > srcSizeSubImage.width || maxSizeY > srcSizeSubImage.height)
            // {
            //     return RPP_ERROR;
            // }

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for (int shadow = 0; shadow < numberOfShadows; shadow++)
            {
                shadowSize.height = rand() % maxSizeY;
                shadowSize.width = rand() % maxSizeX;
                Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
                Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);
                Rpp32u remainingElementsInRow = batch_srcSizeMax[batchCount].width - shadowSize.width;
                for (int c = 0; c < channel; c++)
                {
                    dstPtrTemp = dstPtrSubImage + (c * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ;
                    srcPtrTemp = srcPtrSubImage + (c * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ;

                    for (int i = 0; i < shadowSize.height; i++)
                    {
                        for (int j = 0; j < shadowSize.width; j++)
                        {
                            *dstPtrTemp = *srcPtrTemp / 2;
                            dstPtrTemp++;
                            srcPtrTemp++;
                        }
                        dstPtrTemp += remainingElementsInRow;
                        srcPtrTemp += remainingElementsInRow;
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
            
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u numberOfShadows = batch_numberOfShadows[batchCount];
            Rpp32u maxSizeX = batch_maxSizeX[batchCount];
            Rpp32u maxSizeY = batch_maxSizeY[batchCount];
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
            }

            srand (time(NULL));
            RppiSize srcSizeSubImage, shadowSize;
            T *srcPtrSubImage, *dstPtrSubImage;
            srcSizeSubImage.height = RPPABS(y2 - y1) + 1;
            srcSizeSubImage.width = RPPABS(x2 - x1) + 1;
            srcPtrSubImage = srcPtrImage + (y1 * batch_srcSizeMax[batchCount].width * channel) + (x1 * channel);
            dstPtrSubImage = dstPtrImage + (y1 * batch_srcSizeMax[batchCount].width * channel) + (x1 * channel);

            // if (maxSizeX > srcSizeSubImage.width || maxSizeY > srcSizeSubImage.height)
            // {
            //     std::cerr<<"\n returns error";
            //     return RPP_ERROR;
            // }

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for (int shadow = 0; shadow < numberOfShadows; shadow++)
            {
                // std::cerr<<"\n coming here";
                shadowSize.height = rand() % maxSizeY;
                shadowSize.width = rand() % maxSizeX;
                Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
                Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);
                Rpp32u remainingElementsInRow = channel * (batch_srcSizeMax[batchCount].width - shadowSize.width);
                dstPtrTemp = dstPtrSubImage + (channel * ((shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ));
                srcPtrTemp = srcPtrSubImage + (channel * ((shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ));
                for (int i = 0; i < shadowSize.height; i++)
                {
                    for (int j = 0; j < shadowSize.width; j++)
                    {
                        for (int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = *srcPtrTemp / 2;
                            // *dstPtrTemp = 255;
                            dstPtrTemp++;
                            srcPtrTemp++;
                        }
                    }
                    dstPtrTemp += remainingElementsInRow;
                    srcPtrTemp += remainingElementsInRow;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus random_shadow_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                             Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                             Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, 
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    srand (time(NULL));
    RppiSize srcSizeSubImage, dstSizeSubImage, shadowSize;
    T *srcPtrSubImage, *dstPtrSubImage;
    
    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    if (maxSizeX > srcSizeSubImage.width || maxSizeY > srcSizeSubImage.height)
    {
        return RPP_ERROR;
    }

    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize, &dstSizeSubImage, x1, y1, x2, y2, chnFormat, channel);
    
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    memcpy(dstPtr, srcPtr, channel * srcSize.height * srcSize.width * sizeof(T));

    for (int shadow = 0; shadow < numberOfShadows; shadow++)
    {
        shadowSize.height = rand() % maxSizeY;
        shadowSize.width = rand() % maxSizeX;
        Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
        Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            Rpp32u remainingElementsInRow = srcSize.width - shadowSize.width;
            for (int c = 0; c < channel; c++)
            {
                dstPtrTemp = dstPtrSubImage + (c * srcSize.height * srcSize.width) + (shadowPosI * srcSize.width) + shadowPosJ;
                srcPtrTemp = srcPtrSubImage + (c * srcSize.height * srcSize.width) + (shadowPosI * srcSize.width) + shadowPosJ;

                for (int i = 0; i < shadowSize.height; i++)
                {
                    for (int j = 0; j < shadowSize.width; j++)
                    {
                        *dstPtrTemp = *srcPtrTemp / 2;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                    dstPtrTemp += remainingElementsInRow;
                    srcPtrTemp += remainingElementsInRow;
                }
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            dstPtrTemp = dstPtrSubImage + (channel * ((shadowPosI * srcSize.width) + shadowPosJ));
            srcPtrTemp = srcPtrSubImage + (channel * ((shadowPosI * srcSize.width) + shadowPosJ));
            Rpp32u remainingElementsInRow = channel * (srcSize.width - shadowSize.width);
            for (int i = 0; i < shadowSize.height; i++)
            {
                for (int j = 0; j < shadowSize.width; j++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp / 2;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                }
                dstPtrTemp += remainingElementsInRow;
                srcPtrTemp += remainingElementsInRow;
            }
        }
    }
 
    return RPP_SUCCESS;
}

/**************** Occlusion ***************/

template <typename T>
RppStatus occlusion_host_batch(T* srcPtr1, RppiSize *batch_srcSize1, RppiSize *batch_srcSizeMax1,
                             T* srcPtr2, RppiSize *batch_srcSize2, RppiSize *batch_srcSizeMax2,T* dstPtr,
                             Rpp32u *batch_src1x1, Rpp32u *batch_src1y1, Rpp32u *batch_src1x2, Rpp32u *batch_src1y2, 
                             Rpp32u *batch_src2x1, Rpp32u *batch_src2y1, Rpp32u *batch_src2x2, Rpp32u *batch_src2y2,
                             Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            // Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            // Rpp32u x1 = batch_x1[batchCount];
            // Rpp32u y1 = batch_y1[batchCount];
            // Rpp32u x2 = batch_x2[batchCount];
            // Rpp32u y2 = batch_y2[batchCount];
            // Rpp32u numberOfShadows = batch_numberOfShadows[batchCount];
            // Rpp32u maxSizeX = batch_maxSizeX[batchCount];
            // Rpp32u maxSizeY = batch_maxSizeY[batchCount];
            
            // T *srcPtrImage, *dstPtrImage;
            // Rpp32u loc = 0;
            // compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            // srcPtrImage = srcPtr + loc;
            // dstPtrImage = dstPtr + loc;
            // //#pragma omp parallel for simd
                // #pragma omp parallel for
            // for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            // {
            //     T *srcPtrTemp, *dstPtrTemp;
            //     srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
            //     dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
            //     memcpy(dstPtrTemp, srcPtrTemp, batch_srcSizeMax[batchCount].width * channel * sizeof(T));
            // }

            // srand (time(NULL));
            // RppiSize srcSizeSubImage,shadowSize;
            // T *srcPtrSubImage, *dstPtrSubImage;
            // srcSizeSubImage.height = RPPABS(y2 - y1) + 1;
            // srcSizeSubImage.width = RPPABS(x2 - x1) + 1;
            // srcPtrSubImage = srcPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + (x1);
            // dstPtrSubImage = dstPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + (x1);

            // if (maxSizeX > srcSizeSubImage.width || maxSizeY > srcSizeSubImage.height)
            // {
            //     return RPP_ERROR;
            // }

            // T *srcPtrTemp, *dstPtrTemp;
            // srcPtrTemp = srcPtrImage;
            // dstPtrTemp = dstPtrImage;
            // //#pragma omp parallel for simd
                // #pragma omp parallel for
            // for (int shadow = 0; shadow < numberOfShadows; shadow++)
            // {
            //     shadowSize.height = rand() % maxSizeY;
            //     shadowSize.width = rand() % maxSizeX;
            //     Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
            //     Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);
            //     Rpp32u remainingElementsInRow = batch_srcSizeMax[batchCount].width - shadowSize.width;
            //     for (int c = 0; c < channel; c++)
            //     {
            //         dstPtrTemp = dstPtrSubImage + (c * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ;
            //         srcPtrTemp = srcPtrSubImage + (c * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (shadowPosI * batch_srcSizeMax[batchCount].width) + shadowPosJ;

            //         for (int i = 0; i < shadowSize.height; i++)
            //         {
            //             for (int j = 0; j < shadowSize.width; j++)
            //             {
            //                 *dstPtrTemp = *srcPtrTemp / 2;
            //                 dstPtrTemp++;
            //                 srcPtrTemp++;
            //             }
            //             dstPtrTemp += remainingElementsInRow;
            //             srcPtrTemp += remainingElementsInRow;
            //         }
            //     }
            // }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u src1x1 = batch_src1x1[batchCount];
            Rpp32u src1y1 = batch_src1y1[batchCount];
            Rpp32u src1x2 = batch_src1x2[batchCount];
            Rpp32u src1y2 = batch_src1y2[batchCount];
            Rpp32u src2x1 = batch_src2x1[batchCount];
            Rpp32u src2y1 = batch_src2y1[batchCount];
            Rpp32u src2x2 = batch_src2x2[batchCount];
            Rpp32u src2y2 = batch_src2y2[batchCount];
            T *src1PtrImage,*src2PtrImage, *dstPtrImage;
            Rpp32u loc1 = 0;
            Rpp32u loc2 = 0;
            compute_image_location_host(batch_srcSizeMax1, batchCount, &loc1, channel);
            compute_image_location_host(batch_srcSizeMax2, batchCount, &loc2, channel);
            src1PtrImage = srcPtr1 + loc1;
            dstPtrImage = dstPtr + loc1;
            src2PtrImage = srcPtr1 + loc2;

            Rpp32u elementsInRow1 = channel * batch_srcSize1[batchCount].width;
            Rpp32u elementsInRowMax1 = channel * batch_srcSizeMax1[batchCount].width;
            
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize1[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = src1PtrImage + (i * elementsInRowMax1);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax1);
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow1 * sizeof(T));
            }
            RppiSize srcSize2SubImage, dstSizeSubImage;
            T *srcPtr2SubImage, *dstPtrSubImage;
            srcSize2SubImage.height = RPPABS(src2y2 - src2y1) + 1;
            srcSize2SubImage.width = RPPABS(src2x2 - src2x1) + 1;
            srcPtr2SubImage = src1PtrImage + (src2y1 * batch_srcSizeMax2[batchCount].width) + (src2x1);
            dstSizeSubImage.height = RPPABS(src1y2 - src1y1) + 1;
            dstSizeSubImage.width = RPPABS(src1x2 - src1x1) + 1;
            dstPtrSubImage = dstPtrImage + (src1y1 * batch_srcSizeMax1[batchCount].width) + (src1x1);
            

            Rpp32f hRatio = (((Rpp32f) (dstSizeSubImage.height - 1)) / ((Rpp32f) (srcSize2SubImage.height - 1)));
            Rpp32f wRatio = (((Rpp32f) (dstSizeSubImage.width - 1)) / ((Rpp32f) (srcSize2SubImage.width - 1)));
            Rpp32f srcLocationRow, srcLocationColumn, pixel;
            Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
            T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
            srcPtrTemp = srcPtr2SubImage;
            dstPtrTemp = dstPtrSubImage;
            Rpp32s elementsInRow = batch_srcSizeMax1[batchCount].width * channel;
            Rpp32u remainingElementsInRowDst = (batch_srcSizeMax1[batchCount].width - dstSizeSubImage.width) * channel;
            for (int i = 0; i < dstSizeSubImage.height; i++)
            {
                srcLocationRow = ((Rpp32f) i) / hRatio;
                srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

                if (srcLocationRowFloor > (srcSize2SubImage.height - 2))
                {
                    srcLocationRowFloor = srcSize2SubImage.height - 2;
                }

                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                for (int j = 0; j < dstSizeSubImage.width; j++)
                {   
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    if (srcLocationColumnFloor > (srcSize2SubImage.width - 2))
                    {
                        srcLocationColumnFloor = srcSize2SubImage.width - 2;
                    }

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;
                    
                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                                + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                        
                        *dstPtrTemp = (T) round(pixel);
                        dstPtrTemp ++;
                    }
                }
                dstPtrTemp = dstPtrTemp + remainingElementsInRowDst;
            }
        }
    }
    
    return RPP_SUCCESS;
}



template <typename T>
RppStatus occlusion_host(T* srcPtr1, RppiSize srcSize1, T* srcPtr2, RppiSize srcSize2, T* dstPtr, 
                            Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                            Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, 
                            RppiChnFormat chnFormat, Rpp32u channel)
{
    memcpy(dstPtr, srcPtr1, channel * srcSize1.height * srcSize1.width * sizeof(T));

    RppiSize srcSize2SubImage, dstSizeSubImage;
    T *srcPtr2SubImage, *dstPtrSubImage;

    compute_subimage_location_host(srcPtr2, &srcPtr2SubImage, srcSize2, &srcSize2SubImage, src2x1, src2y1, src2x2, src2y2, chnFormat, channel);
    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize1, &dstSizeSubImage, src1x1, src1y1, src1x2, src1y2, chnFormat, channel);

    Rpp32f hRatio = (((Rpp32f) (dstSizeSubImage.height - 1)) / ((Rpp32f) (srcSize2SubImage.height - 1)));
    Rpp32f wRatio = (((Rpp32f) (dstSizeSubImage.width - 1)) / ((Rpp32f) (srcSize2SubImage.width - 1)));
    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr2SubImage;
    dstPtrTemp = dstPtrSubImage;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRowDst = srcSize1.width - dstSizeSubImage.width;
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr2SubImage + (c * srcSize1.height * srcSize1.width);
            dstPtrTemp = dstPtrSubImage + (c * srcSize1.height * srcSize1.width);
            for (int i = 0; i < dstSizeSubImage.height; i++)
            {   
                srcLocationRow = ((Rpp32f) i) / hRatio;
                srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                if (srcLocationRowFloor > (srcSize2SubImage.height - 2))
                {
                    srcLocationRowFloor = srcSize2SubImage.height - 2;
                }

                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize1.width;
                srcPtrBottomRow  = srcPtrTopRow + srcSize1.width;
                
                for (int j = 0; j < dstSizeSubImage.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                    if (srcLocationColumnFloor > (srcSize2SubImage.width - 2))
                    {
                        srcLocationColumnFloor = srcSize2SubImage.width - 2;
                    }
                    pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));
                    
                    *dstPtrTemp = (T) round(pixel);
                    dstPtrTemp ++;
                }
                dstPtrTemp = dstPtrTemp + remainingElementsInRowDst;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize1.width * channel;
        Rpp32u remainingElementsInRowDst = (srcSize1.width - dstSizeSubImage.width) * channel;
        for (int i = 0; i < dstSizeSubImage.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / hRatio;
            srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

            if (srcLocationRowFloor > (srcSize2SubImage.height - 2))
            {
                srcLocationRowFloor = srcSize2SubImage.height - 2;
            }

            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            for (int j = 0; j < dstSizeSubImage.width; j++)
            {   
                srcLocationColumn = ((Rpp32f) j) / wRatio;
                srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                if (srcLocationColumnFloor > (srcSize2SubImage.width - 2))
                {
                    srcLocationColumnFloor = srcSize2SubImage.width - 2;
                }

                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;
                
                for (int c = 0; c < channel; c++)
                {
                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                    
                    *dstPtrTemp = (T) round(pixel);
                    dstPtrTemp ++;
                }
            }
            dstPtrTemp = dstPtrTemp + remainingElementsInRowDst;
        }
    }
    
    return RPP_SUCCESS;
}

/**************** Jitter ***************/

template <typename T>
RppStatus jitter_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, 
                              Rpp32u *batch_kernelSize,  
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel)
{
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        #pragma omp parallel for
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;
            // std::cerr<<"\n "<<x1<<"\t"<<y1<<"\t"<<x2<<"\t"<<y2;
            Rpp32u kernelSize = batch_kernelSize[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;
            int bound = (kernelSize - 1) / 2;
            srand(time(0)); 
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                //#pragma omp parallel for simd
                #pragma omp parallel for
                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32f pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    if (!((y1 <= i) && (i <= y2)))
                    // if( i <= y1 && i >= y2)
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                        {
                            if((x1 <= j) && (j <= x2 ))
                            // if( j <= x1 && j >= x2)
                            {

                                int pixIdx = i * batch_srcSizeMax[batchCount].width + j;
                                // int channelPixel = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
                                int nhx = rand() % (kernelSize);
                                int nhy = rand() % (kernelSize);
                                if((i - bound + nhy) >= 0 && (i - bound + nhy) <= batch_srcSize[batchCount].height - 1 && (j - bound + nhx) >= 0 && (j - bound + nhx) <= batch_srcSize[batchCount].width - 1)
                                {
                                    int index = ((i - bound) * batch_srcSizeMax[batchCount].width) + (j - bound) + (nhy * batch_srcSizeMax[batchCount].width) + (nhx);
                                    *(dstPtrImage + pixIdx + (batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width * c)) = *(srcPtrImage + index + (batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width * c));  
                                }
                                else 
                                {
                                    *(dstPtrImage + pixIdx + (batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width * c)) = *(srcPtrImage + pixIdx + (batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width * c));  
                                }
                                srcPtrTemp ++; //+= channel;
                                dstPtrTemp ++ ; //channel;
                            }
                            else
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                srcPtrTemp++;
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
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            
            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;

            Rpp32u kernelSize = batch_kernelSize[batchCount];
            
            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            
            int bound = (kernelSize - 1) / 2;
            srand(time(0)); 
            //#pragma omp parallel for simd
                #pragma omp parallel for
            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;

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
                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        if (!((x1 <= j) && (j <= x2 )))
                        {
                            memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));

                            dstPtrTemp += channel;
                            srcPtrTemp += channel;
                        }
                        else
                        {
                            int pixIdx = i * channel * batch_srcSizeMax[batchCount].width + j * channel;
                            int nhx = rand() % (kernelSize);
                            int nhy = rand() % (kernelSize);
                            if((i - bound + nhy) >= 0 && (i - bound + nhy) <= batch_srcSize[batchCount].height - 1 && (j - bound + nhx) >= 0 && (j - bound + nhx) <= batch_srcSize[batchCount].width - 1)
                            {
                                int index = ((i - bound) * channel * batch_srcSizeMax[batchCount].width) + ((j - bound) * channel) + (nhy * channel * batch_srcSizeMax[batchCount].width) + (nhx * channel);
                                for(int k = 0 ; k < channel ; k++)
                                {
                                    *(dstPtrImage + pixIdx + k) = *(srcPtrImage + index + k);  
                                }
                            }
                            else 
                            {
                                for(int k = 0 ; k < channel ; k++)
                                {
                                    *(dstPtrImage + pixIdx + k) = *(srcPtrImage + pixIdx + k);  
                                }
                            }
                            srcPtrTemp += channel;
                            dstPtrTemp += channel;
                        }
                    }
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus jitter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u kernelSize, 
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp8u *dstTemp,*srcTemp;
    dstTemp = dstPtr;
    srcTemp = srcPtr;
    int bound = (kernelSize - 1) / 2;
    srand(time(0)); 
    unsigned int width = srcSize.width;
    unsigned int height = srcSize.height;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        for(int id_y = 0 ; id_y < srcSize.height ; id_y++)
        {
            for(int id_x = 0 ; id_x < srcSize.width ; id_x++)
            {
                int pixIdx = id_y * channel * width + id_x * channel;
                int nhx = rand() % (kernelSize);
                int nhy = rand() % (kernelSize);
                if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
                {
                    int index = ((id_y - bound) * channel * width) + ((id_x - bound) * channel) + (nhy * channel * width) + (nhx * channel);
                    for(int i = 0 ; i < channel ; i++)
                    {
                        *(dstPtr + pixIdx + i) = *(srcPtr + index + i);  
                    }
                }
                else 
                {
                    for(int i = 0 ; i < channel ; i++)
                    {
                        *(dstPtr + pixIdx + i) = *(srcPtr + pixIdx + i);  
                    }
                }
            }
        }
    }
    else
    {
        for(int id_y = 0 ; id_y < srcSize.height ; id_y++)
        {
            for(int id_x = 0 ; id_x < srcSize.width ; id_x++)
            {
                int pixIdx = id_y * width + id_x;
                int channelPixel = height * width;
                int nhx = rand() % (kernelSize);
                int nhy = rand() % (kernelSize);
                int bound = (kernelSize - 1) / 2;
                if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
                {
                    int index = ((id_y - bound) * width) + (id_x - bound) + (nhy * width) + (nhx);
                    for(int i = 0 ; i < channel ; i++)
                    {
                        *(dstPtr + pixIdx + (height * width * i)) = *(srcPtr + index + (height * width * i));  
                    }
                }
                else 
                {
                    for(int i = 0 ; i < channel ; i++)
                    {
                        *(dstPtr + pixIdx + (height * width * i)) = *(srcPtr + pixIdx + (height * width * i));  
                    }
                }
            }
        }
        
    }
    
    
    return RPP_SUCCESS;
}

#endif