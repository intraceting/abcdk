/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/vcodec.h"

#ifdef AVCODEC_AVCODEC_H

static struct _abcdk_torch_vcodec_ffmpeg_dict
{
    /**本地。*/
    int local;

    /**FFMPEG。*/
    int ffmpeg;

} abcdk_torch_vcodec_ffmpeg_dict[] = {
    {ABCDK_TORCH_VCODEC_H264, AV_CODEC_ID_H264},
    {ABCDK_TORCH_VCODEC_HEVC, AV_CODEC_ID_H265},
    {ABCDK_TORCH_VCODEC_MJPEG, AV_CODEC_ID_MJPEG},
    {ABCDK_TORCH_VCODEC_MPEG1VIDEO, AV_CODEC_ID_MPEG1VIDEO},
    {ABCDK_TORCH_VCODEC_MPEG2VIDEO, AV_CODEC_ID_MPEG2VIDEO},
    {ABCDK_TORCH_VCODEC_MPEG4, AV_CODEC_ID_MPEG4},
    {ABCDK_TORCH_VCODEC_VC1, AV_CODEC_ID_VC1},
    {ABCDK_TORCH_VCODEC_VP8, AV_CODEC_ID_VP8},
    {ABCDK_TORCH_VCODEC_VP9, AV_CODEC_ID_VP9},
    {ABCDK_TORCH_VCODEC_WMV3, AV_CODEC_ID_WMV3}};

int abcdk_torch_vcodec_convert_to_ffmpeg(int format)
{
    struct _abcdk_torch_vcodec_ffmpeg_dict *p;

    assert(format >= 0);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_torch_vcodec_ffmpeg_dict); i++)
    {
        p = &abcdk_torch_vcodec_ffmpeg_dict[i];

        if (p->local == format)
            return p->ffmpeg;
    }

    return -1;
}

int abcdk_torch_vcodec_convert_from_ffmpeg(int format)
{
    struct _abcdk_torch_vcodec_ffmpeg_dict *p;

    assert(format >= 0);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_torch_vcodec_ffmpeg_dict); i++)
    {
        p = &abcdk_torch_vcodec_ffmpeg_dict[i];

        if (p->ffmpeg == format)
            return p->local;
    }

    return -1;
}

#else // AVUTIL_PIXFMT_H

int abcdk_torch_vcodec_convert_to_ffmpeg(int format)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

int abcdk_torch_vcodec_convert_from_ffmpeg(int format)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

#endif // AVCODEC_AVCODEC_H