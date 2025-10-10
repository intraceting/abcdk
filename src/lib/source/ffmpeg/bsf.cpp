/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/bsf.h"

struct _abcdk_ffmpeg_bsf
{
    const AVBitStreamFilter *filter_p;
    AVBSFContext *bsf_ctx;
    std::queue<AVPacket *> send_cache;
};// abcdk_ffmpeg_bsf_t;


void abcdk_ffmpeg_bsf_free(abcdk_ffmpeg_bsf_t **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_bsf_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    av_bsf_free(&ctx_p->bsf_ctx);

    while (ctx_p->send_cache.size() > 0)
    {
        av_packet_free(&ctx_p->send_cache.front());
        ctx_p->send_cache.pop();
    }

    // free.
    delete ctx_p;
#endif //#ifndef HAVE_FFMPEG
}

abcdk_ffmpeg_bsf_t *abcdk_ffmpeg_bsf_alloc(const char *name)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_bsf_t *ctx;
    int chk;

    assert(name != NULL);

    ctx = new abcdk_ffmpeg_bsf_t;
    if (!ctx)
        return NULL;

    ctx->filter_p = av_bsf_get_by_name(name);
    if (!ctx->filter_p)
    {
        abcdk_ffmpeg_bsf_free(&ctx);
        return NULL;
    }

    return ctx;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_bsf_init(abcdk_ffmpeg_bsf_t *ctx, const AVCodecContext *codec_ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && codec_ctx != NULL);

    assert(ctx->bsf_ctx == NULL);

    chk = av_bsf_alloc(ctx->filter_p, &ctx->bsf_ctx);
    if (chk < 0)
        return -1;

    chk = avcodec_parameters_from_context(ctx->bsf_ctx->par_in,codec_ctx);
    if (chk < 0)
        return -2;
        
    chk = av_bsf_init(ctx->bsf_ctx);
    if (chk < 0)
        return -3;

    return 0;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_bsf_init2(abcdk_ffmpeg_bsf_t *ctx, const AVCodecParameters *codec_par)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && codec_par != NULL);

    assert(ctx->bsf_ctx == NULL);

    chk = av_bsf_alloc(ctx->filter_p, &ctx->bsf_ctx);
    if (chk < 0)
        return -1;

    chk = avcodec_parameters_copy(ctx->bsf_ctx->par_in, codec_par);
    if (chk < 0)
        return -2;

    chk = av_bsf_init(ctx->bsf_ctx);
    if (chk < 0)
        return -3;

    return 0;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_bsf_recv(abcdk_ffmpeg_bsf_t *ctx, AVPacket *pkt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && pkt != NULL);

    chk = av_bsf_receive_packet(ctx->bsf_ctx, pkt);
    if (chk == 0)
        return 1;
    else if (chk == AVERROR(EAGAIN))
        return 0;

    return -1;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_bsf_send(abcdk_ffmpeg_bsf_t *ctx, AVPacket *pkt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && pkt != NULL);

    // First, process the packets in the send cache queue.
    while (ctx->send_cache.size() > 0)
    {
        auto pkt_p = ctx->send_cache.front();
        chk = av_bsf_send_packet(ctx->bsf_ctx, pkt_p);
        if (chk != 0)
            break;

        // Pop from the queue and release.
        ctx->send_cache.pop();
        av_packet_free(&pkt_p);
    }

    // Next, process the most recent packet.
    if (ctx->send_cache.size() <= 0)
    {
        chk = av_bsf_send_packet(ctx->bsf_ctx, pkt);
        if (chk == 0)
            return 0;
    }

    // Ignore the error type and clone a copy to save at the end of the queue.
    ctx->send_cache.push(av_packet_clone(pkt));
    // Release the referenced object so that external calls perceive it as successful.
    av_packet_unref(pkt);
    return 0;
#endif //#ifndef HAVE_FFMPEG
}
