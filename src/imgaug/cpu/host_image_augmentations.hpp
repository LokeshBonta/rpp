#include <cpu/rpp_cpu_common.hpp>

/************ Blur************/

template <typename T>
RppStatus blur_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev, unsigned int kernelSize,
                    RppiChnFormat chnFormat, unsigned int channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    Rpp32f *kernel = (Rpp32f *)malloc(kernelSize * kernelSize * sizeof(Rpp32f));
    Rpp32f s, sum = 0.0;
    int bound = ((kernelSize - 1) / 2);
    unsigned int c = 0;
    s = 1 / (2 * stdDev * stdDev);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = (1 / M_PI) * (s) * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }
    RppiSize sizeMod;
    sizeMod.width = srcSize.width + (2 * bound);
    sizeMod.height = srcSize.height + (2 * bound);
    Rpp8u *pSrcMod = (Rpp8u *)malloc(sizeMod.width * sizeMod.height * channel * sizeof(Rpp8u));
    int srcLoc = 0, srcModLoc = 0, dstLoc = 0;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < sizeMod.width; i++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += 1;
                }
            }
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int b = 0; b < bound; b++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += 1;
                }
                for (int j = 0; j < srcSize.width; j++)
                {
                    pSrcMod[srcModLoc] = srcPtr[srcLoc];
                    srcModLoc += 1;
                    srcLoc += 1;
                }
                for (int b = 0; b < bound; b++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += 1;
                }
            }
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < sizeMod.width; i++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += 1;
                }
            }
        }
        dstLoc = 0;
        srcModLoc = 0;
        int count = 0;
        float pixel = 0.0;
        int *convLocs = (int *)malloc(kernelSize * kernelSize * sizeof(int));
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    count = 0;
                    pixel = 0.0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        for (int n = 0; n < kernelSize; n++, count++)
                        {
                            convLocs[count] = srcModLoc + (m * sizeMod.width) + n;
                        }
                    }
                    for (int k = 0; k < (kernelSize * kernelSize); k++)
                    {
                        pixel += (kernel[k] * (float)pSrcMod[convLocs[k]]);
                    }
                    pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
                    pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
                    dstPtr[dstLoc] = (Rpp8u) round(pixel);
                    dstLoc += 1;
                    srcModLoc += 1;
                }
                srcModLoc += (kernelSize - 1);
            }
            srcModLoc += ((kernelSize - 1) * sizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int c = 0; c < channel; c++)
        {
            srcModLoc = c;
            srcLoc = c;
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < sizeMod.width; i++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += channel;
                }
            }
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int b = 0; b < bound; b++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += channel;
                }
                for (int j = 0; j < srcSize.width; j++)
                {
                    pSrcMod[srcModLoc] = srcPtr[srcLoc];
                    srcModLoc += channel;
                    srcLoc += channel;
                }
                for (int b = 0; b < bound; b++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += channel;
                }
            }
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < sizeMod.width; i++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += channel;
                }
            }
            
        }
        dstLoc = 0;
        srcModLoc = 0;
        int count = 0;
        float pixel = 0.0;
        int *convLocs = (int *)malloc(kernelSize * kernelSize * sizeof(int));
        for (int c = 0; c < channel; c++)
        {
            srcModLoc = c;
            dstLoc = c;
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    count = 0;
                    pixel = 0.0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        for (int n = 0; n < kernelSize; n++, count++)
                        {
                            convLocs[count] = srcModLoc + (m * sizeMod.width * channel) + (n * channel);
                        }
                    }
                    for (int k = 0; k < (kernelSize * kernelSize); k++)
                    {
                        pixel += (kernel[k] * (float)pSrcMod[convLocs[k]]);
                    }
                    pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
                    pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
                    dstPtr[dstLoc] = (Rpp8u) round(pixel);
                    dstLoc += channel;
                    srcModLoc += channel;
                }
                srcModLoc += ((kernelSize - 1) * channel);
            }
        }
    }
    
    return RPP_SUCCESS;
}

/************ Brightness ************/

template <typename T>
RppStatus brightness_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                            Rpp32f alpha, Rpp32f beta,
                            RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) * alpha + beta;
        pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
        pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

/**************** Contrast ***************/

template <typename T>
RppStatus contrast_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        Rpp32u new_min, Rpp32u new_max,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            Rpp32f Min, Max;
            Min = srcPtr[c * srcSize.height * srcSize.width];
            Max = srcPtr[c * srcSize.height * srcSize.width];
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                if (srcPtr[i + (c * srcSize.height * srcSize.width)] < Min)
                {
                    Min = srcPtr[i + (c * srcSize.height * srcSize.width)];
                }
                if (srcPtr[i + (c * srcSize.height * srcSize.width)] > Max)
                {
                    Max = srcPtr[i + (c * srcSize.height * srcSize.width)];
                }
            }
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                Rpp32f pixel = (Rpp32f) srcPtr[i + (c * srcSize.height * srcSize.width)];
                pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
                pixel = (pixel < (Rpp32f)new_max) ? pixel : ((Rpp32f)new_max);
                pixel = (pixel > (Rpp32f)new_min) ? pixel : ((Rpp32f)new_min);
                dstPtr[i + (c * srcSize.height * srcSize.width)] = (Rpp8u) pixel;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int c = 0; c < channel; c++)
        {
            Rpp32f Min, Max;
            Min = srcPtr[c];
            Max = srcPtr[c];
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                if (srcPtr[(channel * i) + c] < Min)
                {
                    Min = srcPtr[(channel * i) + c];
                }
                if (srcPtr[(channel * i) + c] > Max)
                {
                    Max = srcPtr[(channel * i) + c];
                }
            }
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                Rpp32f pixel = (Rpp32f) srcPtr[(channel * i) + c];
                pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
                pixel = (pixel < (Rpp32f)new_max) ? pixel : ((Rpp32f)new_max);
                pixel = (pixel > (Rpp32f)new_min) ? pixel : ((Rpp32f)new_min);
                dstPtr[(channel * i) + c] = (Rpp8u) pixel;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** Blend ***************/

template <typename T>
RppStatus blend_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr, 
                        Rpp32f alpha, RppiChnFormat chnFormat, 
                        unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        *dstPtr = ((1 - alpha) * (*srcPtr1)) + (alpha * (*srcPtr2));
        srcPtr1++;
        srcPtr2++;
        dstPtr++;
    }  

    return RPP_SUCCESS;  
}

/**************** Add Noise ***************/

//Gaussian host function

template <typename T>
RppStatus noiseAdd_gaussian_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                        RppiNoise noiseType,  RppiGaussParameter *noiseParameter, 
                        RppiChnFormat chnFormat, unsigned int channel)
{
    std::default_random_engine generator;
    std::normal_distribution<>  distribution{noiseParameter->mean, noiseParameter->sigma}; 
    for(int i = 0; i < (srcSize.height * srcSize.width * channel) ; i++)
    {
        Rpp32f pixel = ((Rpp32f) *srcPtr) + ((Rpp32f) distribution(generator));
		*dstPtr = RPPPIXELCHECK(pixel); 
        dstPtr++;
        srcPtr++;       
    }
    return RPP_SUCCESS;
}

//Salt and Pepper Host function

template <typename T>
RppStatus noiseAdd_snp_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                        RppiNoise noiseType,  Rpp32f *noiseParameter, 
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int i;
    for (i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]);
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }

    Rpp32u noiseProbability= (Rpp32u)(*noiseParameter * srcSize.width * srcSize.height * channel );
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(i = 0 ; i < noiseProbability ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp8u newValue = rand()%2 ? 255 : 1;
            for (int c = 0; c < channel; c++)
            {
                dstPtr[(row * srcSize.width) + (column) + (c * srcSize.width * srcSize.height) ] = newValue;
            }
        }        
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(i = 0 ; i < noiseProbability ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp8u newValue = rand()%2 ? 1 : 255;
            for (int c = 0; c < channel; c++)
            {
                dstPtr[(channel * row * srcSize.width) + (column * channel) + c] = newValue;
            }
        }
    }

    return RPP_SUCCESS;
}
/**************** Gamma Correction ***************/

template <typename T>
RppStatus gamma_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f gamma,
                                RppiChnFormat chnFormat,   unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) / 255;
        pixel = pow(pixel, gamma);
        pixel *= 255;
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

/**************** Fog ***************/
template <typename T>
RppStatus fog_host(T* srcPtr, RppiSize srcSize,
                    Rpp32f fogValue,
                    RppiChnFormat chnFormat,   unsigned int channel, T* temp)
{
    if(fogValue<=0)
    {
        for(int i=0;i<srcSize.height*srcSize.width*channel;i++)
        {
            *srcPtr=*temp;
            srcPtr++;
            temp++;
        }
    }
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = 0; i < (srcSize.width * srcSize.height); i++)
        {
            Rpp32f check=srcPtr[i] + srcPtr[i + srcSize.width * srcSize.height] + srcPtr[i + srcSize.width * srcSize.height * 2];
            if(check >= (240*3) && fogValue!=0)
            {            }
            else if(check>=(170*3))
            {
                Rpp32f pixel = ((Rpp32f) srcPtr[i])  * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
                srcPtr[i] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + srcSize.width * srcSize.height]) * (1.5 + fogValue) + (7*fogValue);
                srcPtr[i + srcSize.width * srcSize.height] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + srcSize.width * srcSize.height * 2]) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
                srcPtr[i + srcSize.width * srcSize.height * 2] = RPPPIXELCHECK(pixel);
            }
            else if(check<=(85*3))
            {
                Rpp32f pixel = ((Rpp32f) srcPtr[i]) * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
                srcPtr[i] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + srcSize.width * srcSize.height]) * (1.5 + pow(fogValue,2)) + (130*fogValue);
                srcPtr[i + srcSize.width * srcSize.height] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + srcSize.width * srcSize.height * 2]) * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
                srcPtr[i + srcSize.width * srcSize.height * 2] = RPPPIXELCHECK(pixel);
            }
            else
            {
                Rpp32f pixel = ((Rpp32f) srcPtr[i]) * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
                srcPtr[i] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + srcSize.width * srcSize.height]) * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
                srcPtr[i + srcSize.width * srcSize.height] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + srcSize.width * srcSize.height * 2]) * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
                srcPtr[i + srcSize.width * srcSize.height * 2] = RPPPIXELCHECK(pixel);
            }
        }
    }
    else
    {
        for (int i = 0; i < (srcSize.width * srcSize.height * channel); i+=3)
        {
            Rpp32f check=srcPtr[i] + srcPtr[i+1] + srcPtr[i+ 2];
            if(check >= (240*3) && fogValue!=0)
            {            }
            else if(check>=(170*3) && fogValue!=0)
            {
                Rpp32f pixel = ((Rpp32f) srcPtr[i]) * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
                srcPtr[i] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + 1]) * (1.5 + fogValue) + (7*fogValue);
                srcPtr[i+1] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + 2]) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
                srcPtr[i+2] = RPPPIXELCHECK(pixel);
            }
            else if(check<=(85*3) && fogValue!=0)
            {
                Rpp32f pixel = ((Rpp32f) srcPtr[i]) * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
                srcPtr[i] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + 1]) * (1.5 + pow(fogValue,2)) + (130*fogValue);
                srcPtr[i+1] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + 2]) * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
                srcPtr[i+2] = RPPPIXELCHECK(pixel);
            }
            else if(fogValue!=0)
            {
                Rpp32f pixel = ((Rpp32f) srcPtr[i]) * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
                srcPtr[i] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + 1]) * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
                srcPtr[i+1] = RPPPIXELCHECK(pixel);
                pixel = ((Rpp32f) srcPtr[i + 2]) * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
                srcPtr[i+2] = RPPPIXELCHECK(pixel);
            }
        }
    }
    return RPP_SUCCESS;

}

/**************** Rain ***************/
template <typename T>
RppStatus rain_host(T* srcPtr, RppiSize srcSize,T* dstPtr,
                    Rpp32f rainValue, Rpp32f rainWidth, Rpp32f rainHeight,
                    RppiChnFormat chnFormat,   unsigned int channel)
{ 
    rainValue=rainValue/10;
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]);
        dstPtr[i] = RPPPIXELCHECK(pixel);
    }

    Rpp32u rainProbability= (Rpp32u)(rainValue * srcSize.width * srcSize.height * channel );
    
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int i = 0 ; i < rainProbability ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k=0;k<channel;k++)
            {
                pixel=(Rpp32f)dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width)] + 5;
                dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width)] = RPPPIXELCHECK(pixel) ;
            }
            if (row+rainHeight < srcSize.height && column+rainWidth< srcSize.width)
            {
                for(int j=1;j<rainHeight;j++)
                {
                    for(int k=0;k<channel;k++)
                    {
                        for(int m=0;m<rainWidth;m++)
                        {
                            pixel=(Rpp32f)dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width) + (srcSize.width*j)+m]+5;
                            dstPtr[(row * srcSize.width) + (column) + (k*srcSize.height*srcSize.width) + (srcSize.width*j)+m] = RPPPIXELCHECK(pixel) ;
                        }
                    }            
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0 ; i < rainProbability ; i++)
        {
            Rpp32u row = rand() % srcSize.height;
            Rpp32u column = rand() % srcSize.width;
            Rpp32f pixel;
            for(int k=0;k<channel;k++)
            {
                pixel=(Rpp32f)dstPtr[(channel * row * srcSize.width) + (column * channel) + k] + 5;
                dstPtr[(channel * row * srcSize.width) + (column * channel) + k] = RPPPIXELCHECK(pixel) ;
            }
            if (row+rainHeight < srcSize.height && column+rainWidth< srcSize.width)
            {
                for(int j=1;j<rainHeight;j++)
                {
                    for(int k=0;k<channel;k++)
                    {
                        for(int m=0;m<rainWidth;m++)
                        {
                            pixel=(Rpp32f)dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j)+(channel*m)]+5;
                            dstPtr[(channel * row * srcSize.width) + (column * channel) + k + (channel * srcSize.width * j)+(channel*m)] = RPPPIXELCHECK(pixel) ;
                        } 
                    }            
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}