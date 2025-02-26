/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/cdcfmt.h"

#ifdef AVCODEC_AVCODEC_H

static struct _abcdk_media_cdcfmt_ffmpeg_dict
{
    /**本地。*/
    int local;

    /**FFMPEG。*/
    int ffmpeg;

} abcdk_media_cdcfmt_ffmpeg_dict[] = {
    {ABCDK_MEDIA_CDCFMT_H264, AV_CODEC_ID_H264},
    {ABCDK_MEDIA_CDCFMT_HEVC, AV_CODEC_ID_H265},
    {ABCDK_MEDIA_CDCFMT_MJPEG, AV_CODEC_ID_MJPEG},
    {ABCDK_MEDIA_CDCFMT_MPEG1VIDEO, AV_CODEC_ID_MPEG1VIDEO},
    {ABCDK_MEDIA_CDCFMT_MPEG2VIDEO, AV_CODEC_ID_MPEG2VIDEO},
    {ABCDK_MEDIA_CDCFMT_MPEG4, AV_CODEC_ID_MPEG4},
    {ABCDK_MEDIA_CDCFMT_VC1, AV_CODEC_ID_VC1},
    {ABCDK_MEDIA_CDCFMT_VP8, AV_CODEC_ID_VP8},
    {ABCDK_MEDIA_CDCFMT_VP9, AV_CODEC_ID_VP9},
    {ABCDK_MEDIA_CDCFMT_WMV3, AV_CODEC_ID_WMV3}};

int abcdk_media_cdcfmt_to_ffmpeg(int cdcfmt)
{
    struct _abcdk_media_cdcfmt_ffmpeg_dict *p;

    assert(cdcfmt > 0);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_media_cdcfmt_ffmpeg_dict); i++)
    {
        p = &abcdk_media_cdcfmt_ffmpeg_dict[i];

        if (p->local == cdcfmt)
            return p->ffmpeg;
    }

    return -1;
}

int abcdk_media_cdcfmt_from_ffmpeg(int cdcfmt)
{
    struct _abcdk_media_cdcfmt_ffmpeg_dict *p;

    assert(cdcfmt > 0);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_media_cdcfmt_ffmpeg_dict); i++)
    {
        p = &abcdk_media_cdcfmt_ffmpeg_dict[i];

        if (p->ffmpeg == cdcfmt)
            return p->local;
    }

    return -1;
}

#else // AVUTIL_PIXFMT_H

int abcdk_media_cdcfmt_to_ffmpeg(int cdcfmt)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

int abcdk_media_cdcfmt_from_ffmpeg(int cdcfmt)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

#endif // AVCODEC_AVCODEC_H