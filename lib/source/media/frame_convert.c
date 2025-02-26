/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/frame.h"

#ifdef SWSCALE_SWSCALE_H

int abcdk_media_frame_convert(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    struct SwsContext *ctx = NULL;
    AVFrame tmp_dst = {0}, tmp_src = {0};

    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_MEDIA_TAG_HOST);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST);

    for (int i = 0; i < 4; i++)
    {
        tmp_dst.data[i] = dst->data[i];
        tmp_dst.linesize[i] = dst->stride[i];
        tmp_src.data[i] = src->data[i];
        tmp_src.linesize[i] = src->stride[i];
    }

    tmp_dst.format = abcdk_media_pixfmt_to_ffmpeg(dst->pixfmt);
    tmp_dst.width = dst->width;
    tmp_dst.height = dst->height;

    tmp_src.format = abcdk_media_pixfmt_to_ffmpeg(src->pixfmt);
    tmp_src.width = src->width;
    tmp_src.height = src->height;

    ctx = abcdk_sws_alloc2(tmp_src, tmp_dst, 0);
    if (!ctx)
        return -1;

    chk = abcdk_sws_scale(ctx, tmp_src, tmp_dst);
    abcdk_sws_free(&ctx);

    return -1;
}

#else

int abcdk_media_frame_convert(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    return -1;
}

#endif //SWSCALE_SWSCALE_H