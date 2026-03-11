/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_PIXFMT_HXX
#define ABCDK_XPU_NVIDIA_PIXFMT_HXX

#include "abcdk/xpu/pixfmt.h"
#include "../base.in.h"
#include "../common/pixfmt.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace pixfmt
        {
            static inline AVPixelFormat local_to_ffmpeg(abcdk_xpu_pixfmt_t format)
            {
                return common::pixfmt::local_to_ffmpeg(format);
            }

            static inline AVPixelFormat local_to_ffmpeg(int format)
            {
                return local_to_ffmpeg((abcdk_xpu_pixfmt_t)format);
            }
            
            static inline abcdk_xpu_pixfmt_t ffmpeg_to_local(AVPixelFormat format)
            {
                return common::pixfmt::ffmpeg_to_local(format);
            }

            static inline abcdk_xpu_pixfmt_t ffmpeg_to_local(int format)
            {
                return ffmpeg_to_local((AVPixelFormat)format);
            }

            static inline int get_bit(abcdk_xpu_pixfmt_t pixfmt, int have_pad)
            {
                return common::pixfmt::get_bit(pixfmt, have_pad);
            }

            static inline const char *get_name(abcdk_xpu_pixfmt_t pixfmt)
            {
                return common::pixfmt::get_name(pixfmt);
            }

            static inline int get_channel(abcdk_xpu_pixfmt_t pixfmt)
            {
                return common::pixfmt::get_channel(pixfmt);
            }

        } // namespace pixfmt
    } // namespace nvidia

} // namespace abcdk_xpu

#endif //ABCDK_XPU_NVIDIA_PIXFMT_HXX