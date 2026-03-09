/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/util.h"
#include "pixfmt.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace pixfmt
        {
            static struct _ffmpeg_dict
            {
                /**本地.*/
                abcdk_xpu_pixfmt_t local;

                /**FFMPEG.*/
                AVPixelFormat ffmpeg;

            } ffmpeg_dict[] = {
                {ABCDK_XPU_PIXFMT_YUV420P, AV_PIX_FMT_YUV420P},
                {ABCDK_XPU_PIXFMT_YUV420P, AV_PIX_FMT_YUVJ420P}, // Same as above
                {ABCDK_XPU_PIXFMT_YUV420P9, AV_PIX_FMT_YUV420P9},
                {ABCDK_XPU_PIXFMT_YUV420P10, AV_PIX_FMT_YUV420P10},
                {ABCDK_XPU_PIXFMT_YUV420P12, AV_PIX_FMT_YUV420P12},
                {ABCDK_XPU_PIXFMT_YUV420P14, AV_PIX_FMT_YUV420P14},
                {ABCDK_XPU_PIXFMT_YUV420P16, AV_PIX_FMT_YUV420P16},
                {ABCDK_XPU_PIXFMT_YUV422P, AV_PIX_FMT_YUV422P},
                {ABCDK_XPU_PIXFMT_YUV422P9, AV_PIX_FMT_YUV422P9},
                {ABCDK_XPU_PIXFMT_YUV422P10, AV_PIX_FMT_YUV422P10},
                {ABCDK_XPU_PIXFMT_YUV422P12, AV_PIX_FMT_YUV422P12},
                {ABCDK_XPU_PIXFMT_YUV422P14, AV_PIX_FMT_YUV422P14},
                {ABCDK_XPU_PIXFMT_YUV422P16, AV_PIX_FMT_YUV422P16},
                {ABCDK_XPU_PIXFMT_YUV444P, AV_PIX_FMT_YUV444P},
                {ABCDK_XPU_PIXFMT_YUV444P9, AV_PIX_FMT_YUV444P9},
                {ABCDK_XPU_PIXFMT_YUV444P10, AV_PIX_FMT_YUV444P10},
                {ABCDK_XPU_PIXFMT_YUV444P12, AV_PIX_FMT_YUV444P12},
                {ABCDK_XPU_PIXFMT_YUV444P14, AV_PIX_FMT_YUV444P14},
                {ABCDK_XPU_PIXFMT_YUV444P16, AV_PIX_FMT_YUV444P16},
                {ABCDK_XPU_PIXFMT_NV12, AV_PIX_FMT_NV12},
                {ABCDK_XPU_PIXFMT_P016, AV_PIX_FMT_P016},
                {ABCDK_XPU_PIXFMT_NV16, AV_PIX_FMT_NV16},
                {ABCDK_XPU_PIXFMT_NV21, AV_PIX_FMT_NV21},
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 31, 100)
                {ABCDK_XPU_PIXFMT_NV24, AV_PIX_FMT_NV24},
                {ABCDK_XPU_PIXFMT_NV42, AV_PIX_FMT_NV42},
#else  //  LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 31, 100)
                {ABCDK_XPU_PIXFMT_NV24, AV_PIX_FMT_NONE},
                {ABCDK_XPU_PIXFMT_NV42, AV_PIX_FMT_NONE},
#endif // LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 31, 100)
                {ABCDK_XPU_PIXFMT_GRAY8, AV_PIX_FMT_GRAY8},
                {ABCDK_XPU_PIXFMT_GRAY16, AV_PIX_FMT_GRAY16},
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 22, 100)
                {ABCDK_XPU_PIXFMT_GRAYF32, AV_PIX_FMT_GRAYF32},
#else  //  LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 22, 100)
                {ABCDK_XPU_PIXFMT_GRAYF32, -1},
#endif // LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 22, 100)
                {ABCDK_XPU_PIXFMT_RGB24, AV_PIX_FMT_RGB24},
                {ABCDK_XPU_PIXFMT_BGR24, AV_PIX_FMT_BGR24},
                {ABCDK_XPU_PIXFMT_RGB32, AV_PIX_FMT_RGB32},
                {ABCDK_XPU_PIXFMT_BGR32, AV_PIX_FMT_BGR32}};

            AVPixelFormat local_to_ffmpeg(abcdk_xpu_pixfmt_t format)
            {
                struct _ffmpeg_dict *p;

                assert(format > ABCDK_XPU_PIXFMT_NONE);

                for (int i = 0; i < ABCDK_ARRAY_SIZE(ffmpeg_dict); i++)
                {
                    p = &ffmpeg_dict[i];

                    if (p->local == format)
                        return p->ffmpeg;
                }

                return AV_PIX_FMT_NONE;
            }

            abcdk_xpu_pixfmt_t ffmpeg_to_local(AVPixelFormat format)
            {
                struct _ffmpeg_dict *p;

                assert(format > AV_PIX_FMT_NONE);

                for (int i = 0; i < ABCDK_ARRAY_SIZE(ffmpeg_dict); i++)
                {
                    p = &ffmpeg_dict[i];

                    if (p->ffmpeg == format)
                        return p->local;
                }

                return ABCDK_XPU_PIXFMT_NONE;
            }

            int get_bit(abcdk_xpu_pixfmt_t pixfmt, int have_pad)
            {
                AVPixelFormat ff_pixfmt = local_to_ffmpeg(pixfmt);

                if (ff_pixfmt <= AV_PIX_FMT_NONE)
                    return -1;

                return abcdk_ffmpeg_pixfmt_get_bit(ff_pixfmt, have_pad);
            }

            const char *get_name(abcdk_xpu_pixfmt_t pixfmt)
            {
                AVPixelFormat ff_pixfmt = local_to_ffmpeg(pixfmt);

                if (ff_pixfmt <= AV_PIX_FMT_NONE)
                    return NULL;

                return abcdk_ffmpeg_pixfmt_get_name(ff_pixfmt);
            }

            int get_channel(abcdk_xpu_pixfmt_t pixfmt)
            {
                AVPixelFormat ff_pixfmt = local_to_ffmpeg(pixfmt);

                if (ff_pixfmt <= AV_PIX_FMT_NONE)
                    return -1;

                return abcdk_ffmpeg_pixfmt_get_channel(ff_pixfmt);
            }
        } // namespace pixfmt
    } // namespace common
} // namespace abcdk_xpu