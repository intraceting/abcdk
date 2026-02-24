/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGCODEC_HXX
#define ABCDK_XPU_COMMON_IMGCODEC_HXX

#include "abcdk/xpu/imgcodec.h"
#include "../runtime.in.h"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgcodec
        {
            /**
             * @note 仅支持BGR.
             */
            int encode(const cv::Mat &src, std::vector<uint8_t> &dst, const char *ext);
            int encode(const AVFrame *src, std::vector<uint8_t> &dst, const char *ext);

            /**
             * @note 仅支持BGR.
             */
            int decode(const std::vector<uint8_t> &src, cv::Mat &dst);
            int decode(const void *src_data, size_t src_size, cv::Mat &dst);

        } // namespace imgcodec
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGCODEC_HXX