/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_INTER_MODE_HXX
#define ABCDK_TORCH_HOST_INTER_MODE_HXX

#include "abcdk/torch/torch.h"
#include "abcdk/torch/opencv.h"

#ifdef OPENCV_IMGPROC_HPP

namespace abcdk
{
    namespace torch_host
    {
        namespace inter_mode
        {
            static inline cv::InterpolationFlags convert2opencv(int mode)
            {
                if (mode == ABCDK_TORCH_INTER_NEAREST)
                    return cv::INTER_NEAREST;
                else if (mode == ABCDK_TORCH_INTER_LINEAR)
                    return cv::INTER_LINEAR;
                else if (mode == ABCDK_TORCH_INTER_CUBIC)
                    return cv::INTER_CUBIC;
                else
                    return cv::INTER_MAX;
            }

        } // namespace inter_mode

    } // namespace torch_host
} // namespace abcdk

#endif //OPENCV_IMGPROC_HPP

#endif // ABCDK_TORCH_HOST_INTER_MODE_HXX