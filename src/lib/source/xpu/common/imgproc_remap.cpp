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
            int remap(const cv::Mat &src, cv::Mat &dst,
                      const cv::Mat &xmap, const cv::Mat &ymap,
                      cv::InterpolationFlags inter_mode)
            {
                cv::remap(src, dst, xmap, ymap, inter_mode);

                return 0;
            }
        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu
