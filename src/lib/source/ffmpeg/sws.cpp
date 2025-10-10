/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/sws.h"


struct _abcdk_ffmpeg_sws
{
    struct SwsContext *sws_ctx;
    int src_width;
    int src_height;
    AVPixelFormat src_pixfmt;
    int dst_width;
    int dst_height;
    AVPixelFormat dst_pixfmt;
    int flags;
};// abcdk_ffmpeg_sws_t;

#ifdef HAVE_FFMPEG

static int _abcdk_ffmpeg_sws_init(abcdk_ffmpeg_sws_t *ctx, int src_width, int src_height, AVPixelFormat src_pixfmt,
                                  int dst_width, int dst_height, AVPixelFormat dst_pixfmt, int flags)
{
    if (ctx->src_width != src_width || ctx->src_height != src_height || ctx->src_pixfmt != src_pixfmt ||
        ctx->dst_width != dst_width || ctx->dst_height != dst_height || ctx->dst_pixfmt != dst_pixfmt ||
        ctx->flags != flags ||
        ctx->sws_ctx == NULL)
    {
        if (ctx->sws_ctx)
            sws_freeContext(ctx->sws_ctx);

        ctx->sws_ctx = sws_getContext(ctx->src_width = src_width, ctx->src_height = src_height, ctx->src_pixfmt = src_pixfmt,
                                      ctx->dst_width = dst_width, ctx->dst_height = dst_height, ctx->dst_pixfmt = dst_pixfmt,
                                      ctx->flags = flags, NULL, NULL, NULL);
    }

    if (!ctx->sws_ctx)
        return -1;

    return 0;
}

static int _abcdk_ffmpeg_sws_init2(abcdk_ffmpeg_sws_t *ctx ,const AVFrame *src, const AVFrame *dst, int flags)
{
    return _abcdk_ffmpeg_sws_init(ctx ,src->width, src->height, (AVPixelFormat)src->format,
                                  dst->width, dst->height, (AVPixelFormat)dst->format,
                                  flags);
}

#endif //#ifdef HAVE_FFMPEG

void abcdk_ffmpeg_sws_free(abcdk_ffmpeg_sws_t **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_sws_t *ctx_p;
    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->sws_ctx)
        sws_freeContext(ctx_p->sws_ctx);

    // free.
    delete ctx_p;
#endif //#ifndef HAVE_FFMPEG
}

abcdk_ffmpeg_sws_t *abcdk_ffmpeg_sws_alloc()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_sws_t *ctx;

    ctx = new abcdk_ffmpeg_sws_t;
    if (!ctx)
        return NULL;

    ctx->sws_ctx = NULL;
    ctx->src_height = ctx->src_width = -1;
    ctx->dst_height = ctx->dst_width = -1;
    ctx->dst_pixfmt = ctx->src_pixfmt = AV_PIX_FMT_NONE;
    ctx->flags = 0;

    return ctx;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_sws_scale(abcdk_ffmpeg_sws_t *ctx, const AVFrame *src, AVFrame *dst)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && src != NULL && dst != NULL);

    chk = _abcdk_ffmpeg_sws_init2(ctx, src, dst, SWS_BICUBIC);
    if (chk != 0)
        return -1;

    chk = sws_scale(ctx->sws_ctx, (const uint8_t *const *)src->data, src->linesize, 0, src->height, dst->data, dst->linesize);
    if (chk < 0)
        return -2;

    return chk;
#endif //#ifndef HAVE_FFMPEG
}
