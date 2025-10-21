/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/encoder.h"

struct _abcdk_ffmpeg_encoder
{
    const AVCodec *codec_ctx_p;
    AVDictionary *coder_opt;
    AVCodecContext *coder_ctx;
    abcdk_ffmpeg_sws_t *sws_ctx;
    int64_t send_seqnum;
    std::queue<AVFrame *> send_cache;
    AVFrame *send_convert;
    int recv_eof;
}; // abcdk_ffmpeg_encoder_t;

void abcdk_ffmpeg_encoder_free(abcdk_ffmpeg_encoder_t **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return;
#else  // #ifndef HAVE_FFMPEG
    abcdk_ffmpeg_encoder_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    av_dict_free(&ctx_p->coder_opt);
    abcdk_ffmpeg_codec_free(&ctx_p->coder_ctx);
    abcdk_ffmpeg_sws_free(&ctx_p->sws_ctx);

    ctx_p->send_seqnum = 0;

    while (ctx_p->send_cache.size() > 0)
    {
        av_frame_free(&ctx_p->send_cache.front());
        ctx_p->send_cache.pop();
    }

    av_frame_free(&ctx_p->send_convert);

    // free.
    delete ctx_p;
#endif // #ifndef HAVE_FFMPEG
}

abcdk_ffmpeg_encoder_t *abcdk_ffmpeg_encoder_alloc(const AVCodec *codec_ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
#else  // #ifndef HAVE_FFMPEG
    abcdk_ffmpeg_encoder_t *ctx;

    ctx = new abcdk_ffmpeg_encoder_t;
    if (!ctx)
        return NULL;

    ctx->codec_ctx_p = NULL;
    ctx->coder_ctx = NULL;
    ctx->coder_opt = NULL;
    ctx->sws_ctx = abcdk_ffmpeg_sws_alloc();
    ctx->send_seqnum = 0;
    ctx->send_convert = av_frame_alloc();
    ctx->recv_eof = 0;

    ctx->coder_ctx = avcodec_alloc_context3(codec_ctx);
    if (!ctx->coder_ctx)
    {
        abcdk_ffmpeg_encoder_free(&ctx);
        return NULL;
    }

    // copy.
    ctx->codec_ctx_p = codec_ctx;

    return ctx;
#endif // #ifndef HAVE_FFMPEG
}

abcdk_ffmpeg_encoder_t *abcdk_ffmpeg_encoder_alloc2(const char *codec_name)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
#else  // #ifndef HAVE_FFMPEG
    const AVCodec *codec_ctx_p = NULL;

    assert(codec_name != NULL);

    codec_ctx_p = avcodec_find_encoder_by_name(codec_name);
    if (!codec_ctx_p)
        return NULL;

    return abcdk_ffmpeg_encoder_alloc(codec_ctx_p);
#endif // #ifndef HAVE_FFMPEG
}

abcdk_ffmpeg_encoder_t *abcdk_ffmpeg_encoder_alloc3(AVCodecID codec_id)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
#else  // #ifndef HAVE_FFMPEG
    const AVCodec *codec_ctx_p = NULL;

    assert(codec_id > AV_CODEC_ID_NONE);

    codec_ctx_p = avcodec_find_encoder(codec_id);
    if (!codec_ctx_p)
        return NULL;

    return abcdk_ffmpeg_encoder_alloc(codec_ctx_p);
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_init(abcdk_ffmpeg_encoder_t *ctx,const AVCodecParameters *param,
                              const AVRational *time_base,const AVRational *frame_rate)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    AVDictionary *opts = NULL;
    int chk;

    assert(ctx != NULL && param != NULL);
    assert(ctx->coder_ctx != NULL);

    chk = avcodec_parameters_to_context(ctx->coder_ctx, param);
    if (chk < 0)
        return chk;

    ctx->coder_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (ctx->coder_ctx->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        ctx->coder_ctx->time_base = (time_base ? *time_base : av_make_q(1, 30));
        ctx->coder_ctx->framerate = (frame_rate ? *frame_rate : av_make_q(time_base->den, time_base->num));

        ctx->coder_ctx->gop_size = abcdk_ffmpeg_q2d(ctx->coder_ctx->framerate, 1.0);

        if (param->codec_id == AV_CODEC_ID_MJPEG)
            ctx->coder_ctx->color_range = AVCOL_RANGE_JPEG;

        // 低延迟配置, 直播或喊话时需要.
        if (param->video_delay <= 0)
        {
            ctx->coder_ctx->has_b_frames = 0;
            ctx->coder_ctx->max_b_frames = 0;
            av_dict_set(&opts, "preset", "ultrafast", 0); // 快速编码
            av_dict_set(&opts, "tune", "zerolatency", 0); // 禁用所有缓冲/重排.
            av_dict_set(&opts, "rc-lookahead", "0", 0);   // 不做前瞻分析.
        }
    }
    else if (ctx->coder_ctx->codec_type == AVMEDIA_TYPE_AUDIO)
    {
        ctx->coder_ctx->time_base = (time_base ? *time_base : av_make_q(1, 9600));

        if (ctx->coder_ctx->codec_id == AV_CODEC_ID_AAC)
            av_dict_set(&opts, "strict", "-2", 0);
    }
    else
    {
        ; // nothing.
    }

    return 0;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_open(abcdk_ffmpeg_encoder_t *ctx, const AVDictionary *opt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL);
    assert(ctx->coder_ctx != NULL);

    if (opt)
        av_dict_copy(&ctx->coder_opt, opt, 0);

    chk = avcodec_open2(ctx->coder_ctx, ctx->codec_ctx_p, &ctx->coder_opt);
    if (chk < 0)
        return chk;

    return 0;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_get_param(abcdk_ffmpeg_encoder_t *ctx, AVCodecParameters *param)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && param != NULL);
    assert(ctx->coder_ctx != NULL);

    chk = avcodec_parameters_from_context(param, ctx->coder_ctx);
    if (chk < 0)
        return chk;

    return 0;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_recv(abcdk_ffmpeg_encoder_t *ctx, AVPacket *dst)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && dst != NULL);
    assert(ctx->coder_ctx != NULL);

    av_packet_unref(dst);

    if(ctx->recv_eof)
        return -1;

    chk = avcodec_receive_packet(ctx->coder_ctx, dst);
    if (chk == AVERROR(EAGAIN))
        return 0; // no packet.

    if (chk == AVERROR(EINVAL) || chk == AVERROR_EOF)
    {
        ctx->recv_eof = 1;
        return -1;
    }
        
    if (chk != 0)
        return -1; // error.

    return 1;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_send(abcdk_ffmpeg_encoder_t *ctx, AVFrame *src)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else // #ifndef HAVE_FFMPEG
    int need_convert = 0;
    int chk;

    assert(ctx != NULL);
    assert(ctx->coder_ctx != NULL);

    if (src)
    {
        // 统计帧数.
        ctx->send_seqnum += 1;
        // 填充PTS.
        src->pts = ctx->send_seqnum;
    }

    if (src != NULL && (src->format != (int)ctx->coder_ctx->pix_fmt || src->width != ctx->coder_ctx->width || src->height != ctx->coder_ctx->height))
    {
        need_convert = 1; // 需要转格式.

        // free old.
        av_frame_unref(ctx->send_convert);

        ctx->send_convert->format = (int)ctx->coder_ctx->pix_fmt; // copy from encoder.
        ctx->send_convert->width = ctx->coder_ctx->width;         // copy from encoder.
        ctx->send_convert->height = ctx->coder_ctx->height;       // copy from encoder.
        ctx->send_convert->flags = src->flags;
#ifdef FF_API_FRAME_KEY
        ctx->send_convert->key_frame = src->key_frame;
#endif // FF_API_FRAME_KEY
        ctx->send_convert->pict_type = src->pict_type;

        av_frame_get_buffer(ctx->send_convert, 0);                    // allocate buffer.
        ctx->send_convert->pts = src->pts;                            // copy PTS.
        abcdk_ffmpeg_sws_scale(ctx->sws_ctx, src, ctx->send_convert); // convert format.
    }

    // 优先处理缓存积压的.
    while (ctx->send_cache.size() > 0)
    {
        auto cache_p = ctx->send_cache.front();
        chk = avcodec_send_frame(ctx->coder_ctx, cache_p);
        if (chk != 0)
            break;

        // 从队列中弹出并释放.
        ctx->send_cache.pop();
        av_frame_free(&cache_p);
    }

    auto src_p = (need_convert ? ctx->send_convert : src);

    // 如果缓存为空, 则发送当前帧到编码器.
    if (ctx->send_cache.size() <= 0)
    {
        chk = avcodec_send_frame(ctx->coder_ctx, src_p);
        if (chk == 0)
            return 0;
    }

    // 走到这里, 忽略错误类型, 直接追加到缓存末尾.
    if(src_p)
        ctx->send_cache.push(av_frame_clone(src_p));
        
    return 0;
#endif // #ifndef HAVE_FFMPEG
}
