/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgcodec.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace imgcodec
        {
            image::metadata_t *decode(const void *src, size_t size)
            {
                image::metadata_t *dst;
                cv::Mat tmp_dst;
                int chk;

                chk = common::imgcodec::decode(src, size, tmp_dst);
                if (chk != 0)
                    return NULL;

                return image::clone(ABCDK_XPU_PIXFMT_BGR24,tmp_dst,16);
            }
        } // namespace imgcodec
    } // namespace general

} // namespace abcdk_xpu
