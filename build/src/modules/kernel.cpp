/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "rpp_kernels.h"
#include <algorithm>
#include <map>
#include <hip/rpp/kernel.hpp>
#include <hip/rpp/stringutils.hpp>

namespace rpp {

const std::map<std::string, std::string>& kernels()
{
    static const std::map<std::string, std::string> data{    { "ABSOLUTE_DIFFERENCE", std::string(reinterpret_cast<const char*>(ABSOLUTE_DIFFERENCE), ABSOLUTE_DIFFERENCE_SIZE) },
    { "ACCUMULATE", std::string(reinterpret_cast<const char*>(ACCUMULATE), ACCUMULATE_SIZE) },
    { "ADD", std::string(reinterpret_cast<const char*>(ADD), ADD_SIZE) },
    { "BILATERAL_FILTER", std::string(reinterpret_cast<const char*>(BILATERAL_FILTER), BILATERAL_FILTER_SIZE) },
    { "BITWISE_AND", std::string(reinterpret_cast<const char*>(BITWISE_AND), BITWISE_AND_SIZE) },
    { "BITWISE_NOT", std::string(reinterpret_cast<const char*>(BITWISE_NOT), BITWISE_NOT_SIZE) },
    { "BLEND", std::string(reinterpret_cast<const char*>(BLEND), BLEND_SIZE) },
    { "BOX_FILTER", std::string(reinterpret_cast<const char*>(BOX_FILTER), BOX_FILTER_SIZE) },
    { "BRIGHTNESS_CONTRAST", std::string(reinterpret_cast<const char*>(BRIGHTNESS_CONTRAST), BRIGHTNESS_CONTRAST_SIZE) },
    { "CANNY_EDGE_DETECTOR", std::string(reinterpret_cast<const char*>(CANNY_EDGE_DETECTOR), CANNY_EDGE_DETECTOR_SIZE) },
    { "CHANNEL_COMBINE", std::string(reinterpret_cast<const char*>(CHANNEL_COMBINE), CHANNEL_COMBINE_SIZE) },
    { "CHANNEL_EXTRACT", std::string(reinterpret_cast<const char*>(CHANNEL_EXTRACT), CHANNEL_EXTRACT_SIZE) },
    { "COLOR_TEMPERATURE", std::string(reinterpret_cast<const char*>(COLOR_TEMPERATURE), COLOR_TEMPERATURE_SIZE) },
    { "COLORTWIST", std::string(reinterpret_cast<const char*>(COLORTWIST), COLORTWIST_SIZE) },
    { "CONTRAST", std::string(reinterpret_cast<const char*>(CONTRAST), CONTRAST_SIZE) },
    { "CONVERT_BIT_DEPTH", std::string(reinterpret_cast<const char*>(CONVERT_BIT_DEPTH), CONVERT_BIT_DEPTH_SIZE) },
    { "CONVOLUTION", std::string(reinterpret_cast<const char*>(CONVOLUTION), CONVOLUTION_SIZE) },
    { "CROP_MIRROR_NORMALIZE", std::string(reinterpret_cast<const char*>(CROP_MIRROR_NORMALIZE), CROP_MIRROR_NORMALIZE_SIZE) },
    { "CUSTOM_CONVOLUTION", std::string(reinterpret_cast<const char*>(CUSTOM_CONVOLUTION), CUSTOM_CONVOLUTION_SIZE) },
    { "DILATE", std::string(reinterpret_cast<const char*>(DILATE), DILATE_SIZE) },
    { "DUMMY", std::string(reinterpret_cast<const char*>(DUMMY), DUMMY_SIZE) },
    { "ERODE", std::string(reinterpret_cast<const char*>(ERODE), ERODE_SIZE) },
    { "EXCLUSIVE_OR", std::string(reinterpret_cast<const char*>(EXCLUSIVE_OR), EXCLUSIVE_OR_SIZE) },
    { "EXPOSURE", std::string(reinterpret_cast<const char*>(EXPOSURE), EXPOSURE_SIZE) },
    { "FAST_CORNER_DETECTOR", std::string(reinterpret_cast<const char*>(FAST_CORNER_DETECTOR), FAST_CORNER_DETECTOR_SIZE) },
    { "FISH_EYE", std::string(reinterpret_cast<const char*>(FISH_EYE), FISH_EYE_SIZE) },
    { "FLIP", std::string(reinterpret_cast<const char*>(FLIP), FLIP_SIZE) },
    { "FOG", std::string(reinterpret_cast<const char*>(FOG), FOG_SIZE) },
    { "GAMMA_CORRECTION", std::string(reinterpret_cast<const char*>(GAMMA_CORRECTION), GAMMA_CORRECTION_SIZE) },
    { "GAUSSIAN_FILTER", std::string(reinterpret_cast<const char*>(GAUSSIAN_FILTER), GAUSSIAN_FILTER_SIZE) },
    { "GAUSSIAN_IMAGE_PYRAMID", std::string(reinterpret_cast<const char*>(GAUSSIAN_IMAGE_PYRAMID), GAUSSIAN_IMAGE_PYRAMID_SIZE) },
    { "HARRIS_CORNER_DETECTOR", std::string(reinterpret_cast<const char*>(HARRIS_CORNER_DETECTOR), HARRIS_CORNER_DETECTOR_SIZE) },
    { "HIST", std::string(reinterpret_cast<const char*>(HIST), HIST_SIZE) },
    { "HISTOGRAM", std::string(reinterpret_cast<const char*>(HISTOGRAM), HISTOGRAM_SIZE) },
    { "HSV_KERNELS", std::string(reinterpret_cast<const char*>(HSV_KERNELS), HSV_KERNELS_SIZE) },
    { "HUE", std::string(reinterpret_cast<const char*>(HUE), HUE_SIZE) },
    { "INCLUSIVE_OR", std::string(reinterpret_cast<const char*>(INCLUSIVE_OR), INCLUSIVE_OR_SIZE) },
    { "INTEGRAL", std::string(reinterpret_cast<const char*>(INTEGRAL), INTEGRAL_SIZE) },
    { "JITTER", std::string(reinterpret_cast<const char*>(JITTER), JITTER_SIZE) },
    { "LAPLACIAN_IMAGE_PYRAMID", std::string(reinterpret_cast<const char*>(LAPLACIAN_IMAGE_PYRAMID), LAPLACIAN_IMAGE_PYRAMID_SIZE) },
    { "LENS_CORRECTION", std::string(reinterpret_cast<const char*>(LENS_CORRECTION), LENS_CORRECTION_SIZE) },
    { "LOCAL_BINARY_PATTERN", std::string(reinterpret_cast<const char*>(LOCAL_BINARY_PATTERN), LOCAL_BINARY_PATTERN_SIZE) },
    { "LOOK_UP_TABLE", std::string(reinterpret_cast<const char*>(LOOK_UP_TABLE), LOOK_UP_TABLE_SIZE) },
    { "MAGNITUDE", std::string(reinterpret_cast<const char*>(MAGNITUDE), MAGNITUDE_SIZE) },
    { "MATCH_TEMPLATE", std::string(reinterpret_cast<const char*>(MATCH_TEMPLATE), MATCH_TEMPLATE_SIZE) },
    { "MAX", std::string(reinterpret_cast<const char*>(MAX), MAX_SIZE) },
    { "MEAN_STDDEV", std::string(reinterpret_cast<const char*>(MEAN_STDDEV), MEAN_STDDEV_SIZE) },
    { "MEDIAN_FILTER", std::string(reinterpret_cast<const char*>(MEDIAN_FILTER), MEDIAN_FILTER_SIZE) },
    { "MIN", std::string(reinterpret_cast<const char*>(MIN), MIN_SIZE) },
    { "MIN_MAX", std::string(reinterpret_cast<const char*>(MIN_MAX), MIN_MAX_SIZE) },
    { "MULTIPLY", std::string(reinterpret_cast<const char*>(MULTIPLY), MULTIPLY_SIZE) },
    { "NOISE", std::string(reinterpret_cast<const char*>(NOISE), NOISE_SIZE) },
    { "NON_MAX_SUPPRESSION", std::string(reinterpret_cast<const char*>(NON_MAX_SUPPRESSION), NON_MAX_SUPPRESSION_SIZE) },
    { "OCCLUSION", std::string(reinterpret_cast<const char*>(OCCLUSION), OCCLUSION_SIZE) },
    { "PHASE", std::string(reinterpret_cast<const char*>(PHASE), PHASE_SIZE) },
    { "PIXELATE", std::string(reinterpret_cast<const char*>(PIXELATE), PIXELATE_SIZE) },
    { "RAIN", std::string(reinterpret_cast<const char*>(RAIN), RAIN_SIZE) },
    { "RANDOM_SHADOW", std::string(reinterpret_cast<const char*>(RANDOM_SHADOW), RANDOM_SHADOW_SIZE) },
    { "RECONSTRUCTION_LAPLACIAN_IMAGE_PYRAMID", std::string(reinterpret_cast<const char*>(RECONSTRUCTION_LAPLACIAN_IMAGE_PYRAMID), RECONSTRUCTION_LAPLACIAN_IMAGE_PYRAMID_SIZE) },
    { "REMAP", std::string(reinterpret_cast<const char*>(REMAP), REMAP_SIZE) },
    { "RESIZE", std::string(reinterpret_cast<const char*>(RESIZE), RESIZE_SIZE) },
    { "ROTATE", std::string(reinterpret_cast<const char*>(ROTATE), ROTATE_SIZE) },
    { "SCALE", std::string(reinterpret_cast<const char*>(SCALE), SCALE_SIZE) },
    { "SCAN", std::string(reinterpret_cast<const char*>(SCAN), SCAN_SIZE) },
    { "SNOW", std::string(reinterpret_cast<const char*>(SNOW), SNOW_SIZE) },
    { "SOBEL", std::string(reinterpret_cast<const char*>(SOBEL), SOBEL_SIZE) },
    { "SUBTRACT", std::string(reinterpret_cast<const char*>(SUBTRACT), SUBTRACT_SIZE) },
    { "TEMPERATURE", std::string(reinterpret_cast<const char*>(TEMPERATURE), TEMPERATURE_SIZE) },
    { "TENSOR", std::string(reinterpret_cast<const char*>(TENSOR), TENSOR_SIZE) },
    { "THRESHOLDING", std::string(reinterpret_cast<const char*>(THRESHOLDING), THRESHOLDING_SIZE) },
    { "VIGNETTE", std::string(reinterpret_cast<const char*>(VIGNETTE), VIGNETTE_SIZE) },
    { "WARP_AFFINE", std::string(reinterpret_cast<const char*>(WARP_AFFINE), WARP_AFFINE_SIZE) },
    { "WARP_PERSPECTIVE", std::string(reinterpret_cast<const char*>(WARP_PERSPECTIVE), WARP_PERSPECTIVE_SIZE) }};
    return data;
}

std::string GetKernelSrc(std::string name)
{
    // Use the base name of the string
    int start  = 0;
    auto slash = static_cast<int>(name.find_last_of("/\\"));
    if(slash != std::string::npos)
    {
        start = slash + 1;
    }

    int len = name.size();
    auto ex = static_cast<int>(name.rfind('.'));
    if(ex != std::string::npos)
    {
        len = ex - start;
    }

    auto key = name.substr(start, len);
    std::string skey(key);
    std::map<std::string, std::string>::iterator iter;
    std::map<std::string, std::string> kernelMap = kernels();
    for (iter = kernelMap.begin(); iter != kernelMap.end(); ++iter) 
    	//std::cout<<"Key:"<<iter->first<<std::endl;
    // Convert to uppercase
    std::transform(key.begin(), key.end(), key.begin(), ::toupper);

    auto it = kernels().find(key);
    if(it == kernels().end())
        RPP_THROW("Failed to load kernel source: " + key);

    return it->second;
}

} // namespace rpp
