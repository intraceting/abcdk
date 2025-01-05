/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/ffmpeg/swscale.h"

#ifdef SWSCALE_SWSCALE_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"


void abcdk_sws_free(struct SwsContext **ctx)
{
    if(!ctx)
        return;

    if(*ctx)
        sws_freeContext(*ctx);

    /*Set to NULL(0).*/
    *ctx = NULL;
}

struct SwsContext *abcdk_sws_alloc(int src_width, int src_height, enum AVPixelFormat src_pixfmt,
                                   int dst_width, int dst_height, enum AVPixelFormat dst_pixfmt,
                                   int flags)
{
    assert(src_width > 0 && src_height > 0 && src_pixfmt > AV_PIX_FMT_NONE);
    assert(dst_width > 0 && dst_height > 0 && dst_pixfmt > AV_PIX_FMT_NONE);

    return sws_getContext(src_width, src_height, src_pixfmt,
                          dst_width, dst_height, dst_pixfmt,
                          flags, NULL, NULL, NULL);
}

struct SwsContext *abcdk_sws_alloc2(const AVFrame *src, const AVFrame *dst, int flags)
{
    assert(dst != NULL && src != NULL);

    return abcdk_sws_alloc(src->width, src->height, src->format,
                           dst->width, dst->height, dst->format,
                           flags);
}

int abcdk_sws_scale(struct SwsContext *ctx, const AVFrame *src, AVFrame *dst)
{
    assert(ctx != NULL && dst != NULL && src != NULL);

    return sws_scale(ctx, (const uint8_t *const *)src->data, src->linesize, 0, src->height, dst->data, dst->linesize);
}


#pragma GCC diagnostic pop

#endif //SWSCALE_SWSCALE_H
