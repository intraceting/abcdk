/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/encoder.h"

struct _abcdk_ffmpeg_encoder
{
    AVCodecContext *codec_ctx;
    abcdk_ffmpeg_sws_t *sws_ctx;
    int64_t send_seqnum;
    std::queue<AVFrame *> send_cache;
    AVFrame *send_convert;
}; // abcdk_ffmpeg_encoder_t;

void abcdk_ffmpeg_encoder_free(abcdk_ffmpeg_encoder_t **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_encoder_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->codec_ctx)
    {
#ifdef FF_API_AVCODEC_CLOSE
        avcodec_close(ctx_p->codec_ctx);
#endif // FF_API_AVCODEC_CLOSE
        avcodec_free_context(&ctx_p->codec_ctx);
    }

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
#endif //#ifndef HAVE_FFMPEG
}

abcdk_ffmpeg_encoder_t *abcdk_ffmpeg_encoder_alloc()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_encoder_t *ctx;

    ctx = new abcdk_ffmpeg_encoder_t;
    if (!ctx)
        return NULL;

    ctx->codec_ctx = NULL;
    ctx->sws_ctx = abcdk_ffmpeg_sws_alloc();
    ctx->send_seqnum = 0;
    ctx->send_convert = av_frame_alloc();

    return ctx;
#endif //#ifndef HAVE_FFMPEG
}

#ifdef HAVE_FFMPEG
static void _abcdk_ffmpeg_encoder_venc_set_fps(AVCodecContext *ctx, double fps)
{
#if 1
    /*-------------Copy from OpenCV----begin------------------*/

    int frame_rate = (int)(fps + 0.5);
    int frame_rate_base = 1;
    while (fabs(((double)frame_rate / frame_rate_base) - fps) > 0.001)
    {
        frame_rate_base *= 10;
        frame_rate = (int)(fps * frame_rate_base + 0.5);
    }

    ctx->time_base.den = frame_rate;
    ctx->time_base.num = frame_rate_base;

    /* adjust time base for supported framerates */
    if (ctx->codec && ctx->codec->supported_framerates)
    {
        const AVRational *p = ctx->codec->supported_framerates;
        AVRational req = {frame_rate, frame_rate_base};
        const AVRational *best = NULL;
        AVRational best_error = {INT_MAX, 1};
        for (; p->den != 0; p++)
        {
            AVRational error = av_sub_q(req, *p);
            if (error.num < 0)
                error.num *= -1;
            if (av_cmp_q(error, best_error) < 0)
            {
                best_error = error;
                best = p;
            }
        }

        if (best)
        {
            ctx->time_base.den = best->num;
            ctx->time_base.num = best->den;
        }
    }
    /*-------------Copy from OpenCV-----end---------------*/
#else
    ctx->time_base = (AVRational){1, fps};
#endif

    /*非常重要。*/
    ctx->framerate.den = ctx->time_base.num;
    ctx->framerate.num = ctx->time_base.den;
}
#endif //#ifdef HAVE_FFMPEG

int abcdk_ffmpeg_encoder_init(abcdk_ffmpeg_encoder_t *ctx, const AVCodec *codec_ctx, AVCodecParameters *param, int framerate, int device)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    AVDictionary *opts = NULL;
    int chk;

    assert(ctx != NULL && codec_ctx != NULL && param != NULL);
    assert(codec_ctx->id == param->codec_id);

    ctx->codec_ctx = avcodec_alloc_context3(codec_ctx);
    if (!ctx->codec_ctx)
        return -1;

    if(param->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        assert(framerate > 0 && framerate <= 1000);
        assert(param->width > 0);
        assert(param->height > 0);
        assert(param->bit_rate > 0);
        assert(param->format > (int)AV_PIX_FMT_NONE);

        chk = avcodec_parameters_to_context(ctx->codec_ctx, param);
        if (chk < 0)
            return -2;

        ctx->codec_ctx->gop_size = framerate;
        ctx->codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        ctx->codec_ctx->max_b_frames = 0; // no b frame.

        if (param->codec_id == AV_CODEC_ID_MJPEG)
            ctx->codec_ctx->color_range = AVCOL_RANGE_JPEG;

        // 设置FPS.
        _abcdk_ffmpeg_encoder_venc_set_fps(ctx->codec_ctx, framerate);

#if 0
        //低延迟配置, 直播或喊话时需要.
        av_dict_set(&opts, "preset", "ultrafast", 0);
        av_dict_set(&opts, "tune", "zerolatency", 0);
        av_dict_set(&opts, "bframes", "0", 0);
        av_dict_set(&opts, "rc-lookahead", "0", 0);
#endif

    }
    else
    {
        abcdk_trace_printf(LOG_WARNING, TT("尚未支持的类型(%d)."), param->codec_type);
        return -127;
    }

    chk = avcodec_open2(ctx->codec_ctx, codec_ctx, &opts);
    av_dict_free(&opts);

    if (chk < 0)
        return -3;

    return 0;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_init2(abcdk_ffmpeg_encoder_t *ctx, const char *codec_name, AVCodecParameters *param, int framerate, int device)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    const AVCodec *codec_ctx = NULL;

    assert(codec_name != NULL && param != NULL);

    codec_ctx = avcodec_find_encoder_by_name(codec_name);
    if(!codec_ctx)
        return -1;

    return abcdk_ffmpeg_encoder_init(ctx, codec_ctx, param, framerate, device);
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_init3(abcdk_ffmpeg_encoder_t *ctx, AVCodecID codec_id, AVCodecParameters *param, int framerate, int device)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    const AVCodec *codec_ctx = NULL;

    assert(codec_id > AV_CODEC_ID_NONE && param != NULL);

    codec_ctx = avcodec_find_encoder(codec_id);
    if(!codec_ctx)
        return -1;

    return abcdk_ffmpeg_encoder_init(ctx, codec_ctx, param, framerate, device);
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_get_extradata(abcdk_ffmpeg_encoder_t *ctx, void **data)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    assert(ctx != NULL);
    assert(ctx->codec_ctx != NULL);

    if (data)
        *data = ctx->codec_ctx->extradata;

    return ctx->codec_ctx->extradata_size;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_recv(abcdk_ffmpeg_encoder_t *ctx, AVPacket *dst)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && dst != NULL);

    chk = avcodec_receive_packet(ctx->codec_ctx, dst);
    if (chk == AVERROR(EAGAIN) || chk == AVERROR_EOF)
        return 0; // no packet.

    if (chk != 0)
        return -1; // error.

    return 1;
#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_encoder_send(abcdk_ffmpeg_encoder_t *ctx, AVFrame *src)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int need_convert = 0;
    int chk;

    assert(ctx != NULL);

    if (src)
    {
        // 统计帧数.
        ctx->send_seqnum += 1;
        // 填充PTS.
        src->pts = ctx->send_seqnum;
    }

    if (src != NULL && (src->format != (int)ctx->codec_ctx->pix_fmt || src->width != ctx->codec_ctx->width || src->height != ctx->codec_ctx->height))
    {
        need_convert = 1; // 需要转格式.

        // free old.
        av_frame_unref(ctx->send_convert);

        ctx->send_convert->format = (int)ctx->codec_ctx->pix_fmt; // copy from encoder.
        ctx->send_convert->width = ctx->codec_ctx->width;         // copy from encoder.
        ctx->send_convert->height = ctx->codec_ctx->height;       // copy from encoder.
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
        chk = avcodec_send_frame(ctx->codec_ctx, cache_p);
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
        chk = avcodec_send_frame(ctx->codec_ctx, src_p);
        if (chk == 0)
            return 0;
    }

    // 走到这里, 忽略错误类型, 直接追加到缓存末尾.
    ctx->send_cache.push(av_frame_clone(src_p));
    return 0;
#endif //#ifndef HAVE_FFMPEG
}
