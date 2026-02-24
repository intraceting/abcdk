/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgcodec.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgcodec
        {
            int decode(const std::vector<uint8_t> &src, cv::Mat &dst)
            {
                dst = cv::imdecode(src, cv::IMREAD_COLOR_BGR);
                if (dst.empty())
                    return -1;

                return 0;
            }

            int decode(const void *src_data, size_t src_size, cv::Mat &dst)
            {
                std::vector<uint8_t> src;
                
                src.resize(src_size);
                memcpy(src.data(),src_data,src_size);

                return decode(src,dst);
            }
        } // namespace imgcodec
    } // namespace common
} // namespace abcdk_xpu
