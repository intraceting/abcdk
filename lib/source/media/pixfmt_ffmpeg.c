/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/pixfmt.h"

#ifdef AVUTIL_PIXFMT_H

static struct _abcdk_media_pixfmt_ffmpeg_dict
{
    /**本地。*/
    int local;

    /**FFMPEG。*/
    int ffmpeg;

} abcdk_media_pixfmt_ffmpeg_dict[] = {
    {ABCDK_MEDIA_PIXFMT_YUV420P, AV_PIX_FMT_YUV420P},
    {ABCDK_MEDIA_PIXFMT_YUVJ420P, AV_PIX_FMT_YUVJ420P},
    {ABCDK_MEDIA_PIXFMT_I420, AV_PIX_FMT_YUV420P},
    {ABCDK_MEDIA_PIXFMT_YV12, AV_PIX_FMT_YUV420P},
    {ABCDK_MEDIA_PIXFMT_NV12, AV_PIX_FMT_NV12},
    {ABCDK_MEDIA_PIXFMT_NV21, AV_PIX_FMT_NV21},
    {ABCDK_MEDIA_PIXFMT_YUV422P, AV_PIX_FMT_YUV422P},
    {ABCDK_MEDIA_PIXFMT_YUVJ422P, AV_PIX_FMT_YUVJ422P},
    {ABCDK_MEDIA_PIXFMT_YUYV, AV_PIX_FMT_YVYU422},
    {ABCDK_MEDIA_PIXFMT_UYVY, AV_PIX_FMT_UYVY422},
    {ABCDK_MEDIA_PIXFMT_NV16, AV_PIX_FMT_NV16},
    {ABCDK_MEDIA_PIXFMT_YUV444P, AV_PIX_FMT_YUV444P},
    {ABCDK_MEDIA_PIXFMT_YUVJ444P, AV_PIX_FMT_YUVJ444P},
    {ABCDK_MEDIA_PIXFMT_NV24, AV_PIX_FMT_NV24},
    {ABCDK_MEDIA_PIXFMT_YUV411P, AV_PIX_FMT_YUV411P},
    {ABCDK_MEDIA_PIXFMT_RGB24, AV_PIX_FMT_RGB24},
    {ABCDK_MEDIA_PIXFMT_BGR24, AV_PIX_FMT_BGR24},
    {ABCDK_MEDIA_PIXFMT_RGB32, AV_PIX_FMT_RGB32},
    {ABCDK_MEDIA_PIXFMT_BGR32, AV_PIX_FMT_BGR32},
    {ABCDK_MEDIA_PIXFMT_GRAY8, AV_PIX_FMT_GRAY8},
    {ABCDK_MEDIA_PIXFMT_GRAYF32, AV_PIX_FMT_GRAYF32}};

int abcdk_media_pixfmt_to_ffmpeg(int format)
{
    struct _abcdk_media_pixfmt_ffmpeg_dict *p;

    assert(format > 0);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_media_pixfmt_ffmpeg_dict); i++)
    {
        p = &abcdk_media_pixfmt_ffmpeg_dict[i];

        if (p->local == format)
            return p->ffmpeg;
    }

    return -1;
}

int abcdk_media_pixfmt_from_ffmpeg(int format)
{
    struct _abcdk_media_pixfmt_ffmpeg_dict *p;

    assert(format > 0);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_media_pixfmt_ffmpeg_dict); i++)
    {
        p = &abcdk_media_pixfmt_ffmpeg_dict[i];

        if (p->ffmpeg == format)
            return p->local;
    }

    return -1;
}

#else // AVUTIL_PIXFMT_H

int abcdk_media_pixfmt_to_ffmpeg(int format)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

int abcdk_media_pixfmt_from_ffmpeg(int format)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

#endif // AVUTIL_PIXFMT_H