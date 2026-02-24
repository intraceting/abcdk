/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_PIXFMT_HXX
#define ABCDK_XPU_COMMON_PIXFMT_HXX

#include "abcdk/xpu/pixfmt.h"
#include "../runtime.in.h"

namespace abcdk_xpu
{
    namespace common
    {
        namespace pixfmt
        {
            AVPixelFormat local_to_ffmpeg(abcdk_xpu_pixfmt_t format);

            abcdk_xpu_pixfmt_t ffmpeg_to_local(AVPixelFormat format);

            int get_bit(abcdk_xpu_pixfmt_t pixfmt, int have_pad);

            const char *get_name(abcdk_xpu_pixfmt_t pixfmt);

            int get_channel(abcdk_xpu_pixfmt_t pixfmt);

        } // namespace pixfmt
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_PIXFMT_HXX