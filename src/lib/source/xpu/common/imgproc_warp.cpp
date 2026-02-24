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
            int warp(const cv::Mat &src, cv::Mat &dst, const cv::Mat &coeffs, int warp_mode, cv::InterpolationFlags inter_mode)
            {
                cv::Mat m;

                ABCDK_TRACE_ASSERT(!dst.empty(), ABCDK_GETTEXT("目标空间必须预先申请."));

                if (warp_mode == 1)
                {
                    cv::warpPerspective(src, dst, coeffs, cv::Size(dst.cols, dst.rows), inter_mode, cv::BORDER_TRANSPARENT);
                }
                else if (warp_mode == 2)
                {
                    cv::warpAffine(src, dst, coeffs(cv::Rect(0, 0, 3, 2)).clone(), cv::Size(dst.cols, dst.rows), inter_mode, cv::BORDER_TRANSPARENT);
                }

                return 0;
            }

        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu
