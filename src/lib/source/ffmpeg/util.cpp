/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/util.h"

void abcdk_ffmpeg_library_deinit()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    avformat_network_deinit();
#endif //#ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_library_init()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    avformat_network_init();
    avdevice_register_all();
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    avcodec_register_all();
#endif //#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)

#endif //#ifndef HAVE_FFMPEG
}

#ifdef HAVE_FFMPEG

static void _abcdk_ffmpeg_log_callback(void *opaque, int level, const char *fmt, va_list v)
{
    int type;

    if ((AV_LOG_QUIET == level) || (AV_LOG_PANIC == level) || (AV_LOG_FATAL == level) || (AV_LOG_ERROR == level))
        type = LOG_ERR;
    else if (AV_LOG_WARNING == level)
        type = LOG_WARNING;
    else if (AV_LOG_INFO == level)
        type = LOG_INFO;
    else if (AV_LOG_VERBOSE == level)
        type = LOG_DEBUG;
    else
        return;

    abcdk_trace_vprintf(type, fmt, v);
}

#endif //#ifdef HAVE_FFMPEG

void abcdk_ffmpeg_log_redirect()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    av_log_set_callback(_abcdk_ffmpeg_log_callback);
#endif //#ifndef HAVE_FFMPEG
}


void abcdk_ffmpeg_io_free(AVIOContext **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    AVIOContext *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->buffer)
        av_free(ctx_p->buffer);

#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
    avio_context_free(&ctx_p);
#else //#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
    av_free(ctx_p);
#endif //#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)

#endif //#ifndef HAVE_FFMPEG
}

AVIOContext *abcdk_ffmpeg_io_alloc(int buf_blocks, int write_flag)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
#else //#ifndef HAVE_FFMPEG
    int buf_size = 8 * 4096; /* 4k bytes 的倍数。 */
    void *buf = NULL;
    AVIOContext *ctx = NULL;

    if (buf_blocks > 0)
        buf_size = buf_blocks * 4096;

    buf = av_malloc(buf_size);
    if (!buf)
        return NULL;

    ctx = avio_alloc_context((uint8_t *)buf, buf_size, write_flag, NULL, NULL, NULL, NULL);
    if (!ctx)
    {
        av_freep(&buf);
        return NULL;
    }

    return ctx;
#endif //#ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_media_dump(AVFormatContext *ctx,int output)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    if (!ctx)
        return;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    av_dump_format(ctx, 0, ctx->filename, output);
#else //#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    av_dump_format(ctx, 0, ctx->url, output);
#endif //#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)

#endif //#ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_media_option_dump(AVFormatContext *ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    if (!ctx)
        return;

    if (ctx->av_class)
        av_opt_show2((void *)&ctx->av_class, NULL, -1, 0);
    else
        av_log(NULL, AV_LOG_INFO, "No options for this.\n");

    if (ctx->iformat)
    {
        if (ctx->iformat->priv_class)
            av_opt_show2((void *)&ctx->iformat->priv_class, NULL, -1, 0);
        else
            av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->iformat->long_name ? ctx->iformat->long_name : ctx->iformat->name));
    }
    if (ctx->oformat)
    {
        if (ctx->oformat->priv_class)
            av_opt_show2((void *)&ctx->oformat->priv_class, NULL, -1, 0);
        else
            av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->oformat->long_name ? ctx->oformat->long_name : ctx->oformat->name));
    }
#endif //#ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_media_free(AVFormatContext **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    AVFormatContext *ctx_p = NULL;
    AVIOContext *pb = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*
     * IO对象在以下两个条件下，需要单独释放。
     *
     * 1：自定义环境。
     * 2：输出对象。
     */
    if (ctx_p->flags & AVFMT_FLAG_CUSTOM_IO)
        abcdk_ffmpeg_io_free(&ctx_p->pb);
    else if(ctx_p->oformat && !(ctx_p->oformat->flags & AVFMT_NOFILE))
        avio_closep(&ctx_p->pb);

    if (ctx_p->iformat)
        avformat_close_input(&ctx_p);
    else
        avformat_free_context(ctx_p);
#endif //#ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_codec_option_dump(AVCodec *ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    if (!ctx)
        return;

    if (ctx->priv_class)
        av_opt_show2((void *)&ctx->priv_class, NULL, -1, 0);
    else
        av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->long_name ? ctx->long_name : ctx->name));
#endif //#ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_codec_free(AVCodecContext **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return ;
#else //#ifndef HAVE_FFMPEG
    AVCodecContext *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p)
        avcodec_close(ctx_p);

    avcodec_free_context(&ctx_p);
#endif //#ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_q2d(AVRational r, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return 0.0;
#else //#ifndef HAVE_FFMPEG
    double d = (r.num == 0 || r.den == 0 ? 0. : av_q2d(r));

    return scale * d;
#endif //#ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_stream_timebase_q2d(AVStream *vs_ctx,double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return 0.0;
#else //#ifndef HAVE_FFMPEG
    assert(vs_ctx != NULL);
    
    return abcdk_ffmpeg_q2d(vs_ctx->time_base, scale);
#endif //#ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_stream_duration(AVStream *vs_ctx, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return 0.0;
#else //#ifndef HAVE_FFMPEG
    double sec = 0.0;

    assert(vs_ctx != NULL);

    if (vs_ctx->duration > 0)
        sec = (double)vs_ctx->duration * abcdk_ffmpeg_q2d(vs_ctx->time_base,scale);

    return sec;
#endif //#ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_stream_fps(AVFormatContext *ctx, AVStream *vs_ctx,double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return 0.000001;
#else //#ifndef HAVE_FFMPEG
    double fps = 0.000001;

    assert(vs_ctx != NULL);
    
    if (fps < 0.000025)
        fps = abcdk_ffmpeg_q2d(vs_ctx->r_frame_rate,scale);
    if (fps < 0.000025)
        fps = abcdk_ffmpeg_q2d(vs_ctx->avg_frame_rate,scale);
    if (fps < 0.000025)
#if FF_API_LAVF_AVCTX
        fps = 1.0 / abcdk_ffmpeg_q2d(vs_ctx->codec->time_base,scale);
#else //FF_API_LAVF_AVCTX
        fps = 1.0 / abcdk_ffmpeg_q2d(vs_ctx->time_base,scale);
#endif //FF_API_LAVF_AVCTX

    return ABCDK_CLAMP(fps,(double)0.001,(double)999999999.0);
#endif //#ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_stream_ts2sec(AVStream *vs_ctx, int64_t ts, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -0.000001;
#else //#ifndef HAVE_FFMPEG
    double sec = -0.000001;

    assert(vs_ctx != NULL);

    sec = (double)(ts - vs_ctx->start_time) * abcdk_ffmpeg_q2d(vs_ctx->time_base, 1.0 / scale);

    return sec;
#endif //#ifndef HAVE_FFMPEG
}
