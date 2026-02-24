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
            static inline abcdk_object_t *_encode(const image::metadata_t *src, const char *ext)
            {
                abcdk_object_t *dst;
                std::vector<uint8_t> dst_buf;
                int chk;

                chk = common::imgcodec::encode(src, dst_buf, ext);
                if (chk != 0)
                    return NULL;

                dst = abcdk_object_copyfrom(dst_buf.data(), dst_buf.size());
                if (!dst)
                    return NULL;

                return dst;
            }

            abcdk_object_t *encode(const image::metadata_t *src, const char *ext)
            {
                abcdk_object_t *dst;
                image::metadata_t *src_tmp;
                int chk;

                if (src->format != AV_PIX_FMT_BGR24)
                {
                    src_tmp = image::create(src->width, src->height, ABCDK_XPU_PIXFMT_BGR24, 16);
                    if (!src_tmp)
                        return NULL;

                    chk = imgproc::convert(src, src_tmp);
                    if (chk != 0)
                    {
                        image::free(&src_tmp);
                        return NULL;
                    }

                    dst = encode(src_tmp, ext);
                    image::free(&src_tmp);

                    return dst;
                }

                return _encode(src, ext);
            }

        } // namespace imgcodec
    } // namespace general
} // namespace abcdk_xpu
