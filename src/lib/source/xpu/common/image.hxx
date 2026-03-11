/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMAGE_HXX
#define ABCDK_XPU_COMMON_IMAGE_HXX

#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/util.h"
#include "abcdk/xpu/image.h"
#include "../base.in.h"

namespace abcdk_xpu
{
    namespace common
    {
        namespace image
        {
            static inline void free(AVFrame **ctx)
            {
                av_frame_free(ctx);
            }

            static inline AVFrame *alloc()
            {
                return av_frame_alloc();
            }

            static inline void clear(AVFrame *ctx)
            {
                av_frame_unref(ctx);
            }

            static inline int get_buffer(AVFrame *ctx, int align)
            {
                return av_frame_get_buffer(ctx, align);
            }

            static inline int copy(const AVFrame *src, AVFrame *dst)
            {
                abcdk_ffmpeg_image_copy2(dst, src);
                return 0;
            }

            int copy(const cv::Mat &src, AVFrame *dst);
        } // namespace image
    } // namespace common
} // namespace abcdk_xpu

#endif //ABCDK_XPU_COMMON_IMAGE_HXX