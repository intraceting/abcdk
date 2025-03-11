/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/pixfmt.h"

#ifdef AVUTIL_PIXFMT_H

static struct _abcdk_torch_pixfmt_ffmpeg_dict
{
    /**本地。*/
    int local;

    /**FFMPEG。*/
    int ffmpeg;

} abcdk_torch_pixfmt_ffmpeg_dict[] = {
    {ABCDK_TORCH_PIXFMT_YUV420P, AV_PIX_FMT_YUV420P},
    {ABCDK_TORCH_PIXFMT_YUV420P9, AV_PIX_FMT_YUV420P9},
    {ABCDK_TORCH_PIXFMT_YUV420P10, AV_PIX_FMT_YUV420P10},
    {ABCDK_TORCH_PIXFMT_YUV420P12, AV_PIX_FMT_YUV420P12},
    {ABCDK_TORCH_PIXFMT_YUV420P14, AV_PIX_FMT_YUV420P14},
    {ABCDK_TORCH_PIXFMT_YUV420P16, AV_PIX_FMT_YUV420P16},
    {ABCDK_TORCH_PIXFMT_YUV422P, AV_PIX_FMT_YUV422P},
    {ABCDK_TORCH_PIXFMT_YUV422P9, AV_PIX_FMT_YUV422P9},
    {ABCDK_TORCH_PIXFMT_YUV422P10, AV_PIX_FMT_YUV422P10},
    {ABCDK_TORCH_PIXFMT_YUV422P12, AV_PIX_FMT_YUV422P12},
    {ABCDK_TORCH_PIXFMT_YUV422P14, AV_PIX_FMT_YUV422P14},
    {ABCDK_TORCH_PIXFMT_YUV422P16, AV_PIX_FMT_YUV422P16},
    {ABCDK_TORCH_PIXFMT_YUV444P, AV_PIX_FMT_YUV444P},
    {ABCDK_TORCH_PIXFMT_YUV444P9, AV_PIX_FMT_YUV444P9},
    {ABCDK_TORCH_PIXFMT_YUV444P10, AV_PIX_FMT_YUV444P10},
    {ABCDK_TORCH_PIXFMT_YUV444P12, AV_PIX_FMT_YUV444P12},
    {ABCDK_TORCH_PIXFMT_YUV444P14, AV_PIX_FMT_YUV444P14},
    {ABCDK_TORCH_PIXFMT_YUV444P16, AV_PIX_FMT_YUV444P16},
    {ABCDK_TORCH_PIXFMT_NV12, AV_PIX_FMT_NV12},
    {ABCDK_TORCH_PIXFMT_P016, AV_PIX_FMT_P016},
    {ABCDK_TORCH_PIXFMT_NV16, AV_PIX_FMT_NV16},
    {ABCDK_TORCH_PIXFMT_NV21, AV_PIX_FMT_NV21},
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 31, 100)
    {ABCDK_TORCH_PIXFMT_NV24, AV_PIX_FMT_NV24},
    {ABCDK_TORCH_PIXFMT_NV42, AV_PIX_FMT_NV42},
#else // LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 31, 100)
    {ABCDK_TORCH_PIXFMT_NV24, -1},
    {ABCDK_TORCH_PIXFMT_NV42, -1},
#endif // LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 31, 100)
    {ABCDK_TORCH_PIXFMT_GRAY8, AV_PIX_FMT_GRAY8},
    {ABCDK_TORCH_PIXFMT_GRAY16, AV_PIX_FMT_GRAY16},
    {ABCDK_TORCH_PIXFMT_GRAYF32, AV_PIX_FMT_GRAYF32},
    {ABCDK_TORCH_PIXFMT_RGB24, AV_PIX_FMT_RGB24},
    {ABCDK_TORCH_PIXFMT_BGR24, AV_PIX_FMT_BGR24},
    {ABCDK_TORCH_PIXFMT_RGB32, AV_PIX_FMT_RGB32},
    {ABCDK_TORCH_PIXFMT_BGR32, AV_PIX_FMT_BGR32}};

int abcdk_torch_pixfmt_convert_to_ffmpeg(int format)
{
    struct _abcdk_torch_pixfmt_ffmpeg_dict *p;

    assert(format >= 0);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_torch_pixfmt_ffmpeg_dict); i++)
    {
        p = &abcdk_torch_pixfmt_ffmpeg_dict[i];

        if (p->local == format)
            return p->ffmpeg;
    }

    return -1;
}

int abcdk_torch_pixfmt_convert_from_ffmpeg(int format)
{
    struct _abcdk_torch_pixfmt_ffmpeg_dict *p;

    assert(format >= 0);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_torch_pixfmt_ffmpeg_dict); i++)
    {
        p = &abcdk_torch_pixfmt_ffmpeg_dict[i];

        if (p->ffmpeg == format)
            return p->local;
    }

    return -1;
}

#else // AVUTIL_PIXFMT_H

int abcdk_torch_pixfmt_convert_to_ffmpeg(int format)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return -1;
}

int abcdk_torch_pixfmt_convert_from_ffmpeg(int format)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return -1;
}

#endif // AVUTIL_PIXFMT_H