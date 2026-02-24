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
            int resize(const cv::Mat &src, const abcdk_xpu_rect_t *src_roi, cv::Mat &dst, cv::InterpolationFlags inter_mode)
            {
                cv::Rect tmp_src_roi;

                tmp_src_roi.x = (src_roi ? src_roi->x : 0);
                tmp_src_roi.y = (src_roi ? src_roi->y : 0);
                tmp_src_roi.width = (src_roi ? src_roi->width : src.cols);
                tmp_src_roi.height = (src_roi ? src_roi->height : src.rows);

                cv::resize(src(tmp_src_roi), dst, cv::Size(dst.cols,dst.rows), 0, 0, inter_mode);

                return 0;
            }

        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu
