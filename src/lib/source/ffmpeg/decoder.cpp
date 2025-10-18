/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/decoder.h"

struct _abcdk_ffmpeg_decoder
{
    AVCodecContext *codec_ctx;
    abcdk_ffmpeg_sws_t *sws_ctx;
    AVPixelFormat pixfmt;
    AVFrame *recv_cache;
    AVFrame *recv_cache_hw;
    std::queue<AVPacket *> send_cache;
}; // abcdk_ffmpeg_decoder_t;

void abcdk_ffmpeg_decoder_free(abcdk_ffmpeg_decoder_t **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_decoder_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_ffmpeg_codec_free(&ctx_p->codec_ctx);
    abcdk_ffmpeg_sws_free(&ctx_p->sws_ctx);

    av_frame_free(&ctx_p->recv_cache);
    av_frame_free(&ctx_p->recv_cache_hw);

    while (ctx_p->send_cache.size() > 0)
    {
        av_packet_free(&ctx_p->send_cache.front());
        ctx_p->send_cache.pop();
    }

    // free.
    delete ctx_p;
#endif //#ifndef HAVE_FFMPEG
}

abcdk_ffmpeg_decoder_t *abcdk_ffmpeg_decoder_alloc()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_decoder_t *ctx;

    ctx = new abcdk_ffmpeg_decoder_t;
    if (!ctx)
        return NULL;

    ctx->codec_ctx = NULL;
    ctx->sws_ctx = abcdk_ffmpeg_sws_alloc();
    ctx->pixfmt = AV_PIX_FMT_NONE;
    ctx->recv_cache = av_frame_alloc();
    ctx->recv_cache_hw = av_frame_alloc();

    return ctx;
#endif //#ifndef HAVE_FFMPEG
}

#ifdef HAVE_FFMPEG

static AVPixelFormat _abcdk_ffmpeg_decoder_get_format(AVCodecContext *ctx, const AVPixelFormat *pix_fmts)
{
    abcdk_ffmpeg_decoder_t *ctx_p = (abcdk_ffmpeg_decoder_t *)ctx->opaque;
    const AVPixelFormat *p;

    for (p = pix_fmts; *p != -1; p++)
    {
        if (*p == ctx_p->pixfmt)
        {
            return *p;
        }
    }

    return pix_fmts[0];
}

#endif //#ifdef HAVE_FFMPEG

int abcdk_ffmpeg_decoder_init(abcdk_ffmpeg_decoder_t *ctx, const AVCodec *codec_ctx, AVCodecParameters *param, int device)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    AVDictionary *opts = NULL;
    int chk;

    assert(ctx != NULL && codec_ctx != NULL && param != NULL);

    ctx->codec_ctx = avcodec_alloc_context3(codec_ctx);
    if (!ctx->codec_ctx)
        return AVERROR(ENOMEM);

    chk = avcodec_parameters_to_context(ctx->codec_ctx, param);
    if (chk != 0)
        return chk;

    if (param->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        // copy.
        ctx->pixfmt = (AVPixelFormat)param->format;

        ctx->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P; // must be valid.
        ctx->codec_ctx->get_format = _abcdk_ffmpeg_decoder_get_format;
        ctx->codec_ctx->opaque = ctx;
    }
    else
    {
        ;// nothing.
    }

    chk = avcodec_open2(ctx->codec_ctx, codec_ctx, &opts);
    av_dict_free(&opts);

    if (chk != 0)
        return chk;

    return 0;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_decoder_init2(abcdk_ffmpeg_decoder_t *ctx, const char *codec_name, AVCodecParameters *param, int device)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    const AVCodec *codec_ctx = NULL;

    assert(ctx != NULL && codec_name != NULL && param != NULL);

    codec_ctx = avcodec_find_decoder_by_name(codec_name);
    if(!codec_ctx)
        return AVERROR_DECODER_NOT_FOUND;

    return abcdk_ffmpeg_decoder_init(ctx, codec_ctx, param, device);
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_decoder_init3(abcdk_ffmpeg_decoder_t *ctx, AVCodecID codec_id, AVCodecParameters *param, int device)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    const AVCodec *codec_ctx = NULL;

    assert(ctx != NULL && codec_id > AV_CODEC_ID_NONE && param != NULL);

    codec_ctx = avcodec_find_decoder(codec_id);
    if(!codec_ctx)
        return AVERROR_DECODER_NOT_FOUND;

    return abcdk_ffmpeg_decoder_init(ctx, codec_ctx, param, device);
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_decoder_recv(abcdk_ffmpeg_decoder_t *ctx, AVFrame *dst)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && dst != NULL);

    chk = avcodec_receive_frame(ctx->codec_ctx, ctx->recv_cache);
    if (chk == AVERROR(EAGAIN) || chk == AVERROR_EOF)
        return 0; // no frame.

    if (chk != 0)
        return -1; // error.

    if (ctx->recv_cache->format == AV_PIX_FMT_DRM_PRIME)
    {
        av_frame_move_ref(ctx->recv_cache_hw, ctx->recv_cache);
        av_frame_unref(ctx->recv_cache);
        av_hwframe_transfer_data(ctx->recv_cache, ctx->recv_cache_hw, 0);
        av_frame_unref(ctx->recv_cache_hw);
    }

    if (ctx->pixfmt == AV_PIX_FMT_NONE || ctx->pixfmt == (AVPixelFormat)ctx->recv_cache->format)
    {
        av_frame_move_ref(dst, ctx->recv_cache);
    }
    else
    {
        // need by SWS.
        dst->width = ctx->recv_cache->width;
        dst->height = ctx->recv_cache->height;
        dst->flags = ctx->recv_cache->flags;
#ifdef FF_API_FRAME_KEY
        dst->key_frame = ctx->recv_cache->key_frame;
#endif // FF_API_FRAME_KEY
        dst->pict_type = ctx->recv_cache->pict_type;
        dst->pts = ctx->recv_cache->pts;
        dst->format = (int)ctx->pixfmt;

        av_frame_get_buffer(dst, 0); // allocate buffer.
        abcdk_ffmpeg_sws_scale(ctx->sws_ctx, ctx->recv_cache, dst);
        av_frame_unref(ctx->recv_cache); // unreference.
    }

    return 1; // 1 frame.
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_decoder_send(abcdk_ffmpeg_decoder_t *ctx, AVPacket *src)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL);

    // First, process the packets in the send cache queue.
    while (ctx->send_cache.size() > 0)
    {
        auto cache_p = ctx->send_cache.front();
        chk = avcodec_send_packet(ctx->codec_ctx, cache_p);
        if (chk != 0)
            break;

        // Pop from the queue and release.
        ctx->send_cache.pop();
        av_packet_free(&cache_p);
    }

    // Next, process the most recent packet.
    if (ctx->send_cache.size() <= 0)
    {
        chk = avcodec_send_packet(ctx->codec_ctx, src);
        if (chk == 0)
            return 0;
    }

    // Ignore the error type and clone a copy to save at the end of the queue.
    ctx->send_cache.push(av_packet_clone(src));

    return 0;
#endif //#ifndef HAVE_FFMPEG
}
