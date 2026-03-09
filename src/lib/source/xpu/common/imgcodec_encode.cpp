/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "imgcodec.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgcodec
        {
            int encode(const cv::Mat &src, std::vector<uint8_t> &dst, const char *ext)
            {
                bool bchk;

                if (!ext || abcdk_strcmp(".jpg", ext, 0) == 0 || abcdk_strcmp(".jpeg", ext, 0) == 0)
                {
                    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 100};
                    bchk = cv::imencode(".jpg", src, dst, params);
                    if (!bchk)
                        return -1;

                    return 0;
                }
                else if (abcdk_strcmp(".bmp", ext, 0) == 0 ||
                         abcdk_strcmp(".png", ext, 0) == 0 ||
                         abcdk_strcmp(".tiff", ext, 0) == 0)
                {
                    bchk = cv::imencode(ext, src, dst);
                    if (!bchk)
                        return -1;

                    return 0;
                }

                return -1;
            }

            int encode(const AVFrame *src, std::vector<uint8_t> &dst, const char *ext)
            {
                return encode(util::AVFrame2cvMat(src),dst,ext);
            }

        } // namespace imgcodec
    } // namespace common
} // namespace abcdk_xpu
