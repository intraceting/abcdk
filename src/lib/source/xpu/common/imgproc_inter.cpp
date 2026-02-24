/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            cv::InterpolationFlags inter_local_to_opencv(abcdk_xpu_inter_t mode)
            {
                if (mode == ABCDK_XPU_INTER_NEAREST)
                    return cv::INTER_NEAREST;
                else if (mode == ABCDK_XPU_INTER_LINEAR)
                    return cv::INTER_LINEAR;
                else if (mode == ABCDK_XPU_INTER_CUBIC)
                    return cv::INTER_CUBIC;
                else
                    return cv::INTER_MAX;
            }

        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu
