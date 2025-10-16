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

double abcdk_ffmpeg_stream_fps(AVStream *vs_ctx,double scale)
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
    double sec;

    assert(vs_ctx != NULL);

    sec = (double)(ts - vs_ctx->start_time) * abcdk_ffmpeg_q2d(vs_ctx->time_base, 1.0 / scale);

    return sec;
#endif //#ifndef HAVE_FFMPEG
}

int64_t abcdk_ffmpeg_stream_ts2num(AVStream *vs_ctx, int64_t ts, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int64_t num;
    double sec;

    assert(vs_ctx != NULL);

    sec = abcdk_ffmpeg_stream_ts2sec(vs_ctx, ts, scale);
    num = (int64_t)(abcdk_ffmpeg_stream_fps(vs_ctx, scale) * sec + 0.5);

    return num;
#endif //#ifndef HAVE_FFMPEG
}


int abcdk_ffmpeg_stream_parameters_from_context(AVStream *vs_ctx, const AVCodecContext *codec_ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    assert(vs_ctx != NULL && codec_ctx != NULL);

    /*如果是编码，帧率也一并复制。*/
#if FF_API_LAVF_AVCTX
    if (av_codec_is_encoder(vs_ctx->codec->codec))
#else //FF_API_LAVF_AVCTX
    if (av_codec_is_encoder(codec_ctx->codec))
#endif //FF_API_LAVF_AVCTX
    {
        vs_ctx->time_base = codec_ctx->time_base;
        vs_ctx->avg_frame_rate = vs_ctx->r_frame_rate = codec_ctx->framerate;//av_make_q(codec_ctx->time_base.den, codec_ctx->time_base.num);
#if FF_API_LAVF_AVCTX
        vs_ctx->codec->time_base = codec_ctx->time_base;
        vs_ctx->codec->framerate = codec_ctx->framerate;
#endif //FF_API_LAVF_AVCTX
    }

    /**/
    avcodec_parameters_from_context(vs_ctx->codecpar, codec_ctx);

    /*下面的也要复制，因为一些定制的ffmpeg未完成启用新的参数。*/
#if FF_API_LAVF_AVCTX
    vs_ctx->codec->codec_type = codec_ctx->codec_type;
    vs_ctx->codec->codec_id = codec_ctx->codec_id;
    vs_ctx->codec->codec_tag = codec_ctx->codec_tag;
    vs_ctx->codec->bit_rate = codec_ctx->bit_rate;
    vs_ctx->codec->bits_per_coded_sample = codec_ctx->bits_per_coded_sample;
    vs_ctx->codec->bits_per_raw_sample = codec_ctx->bits_per_raw_sample;
    vs_ctx->codec->profile = codec_ctx->profile;
    vs_ctx->codec->level = codec_ctx->level;
    vs_ctx->codec->flags = codec_ctx->flags;

    switch (codec_ctx->codec_type)
    {
    case AVMEDIA_TYPE_VIDEO:
        vs_ctx->codec->pix_fmt = codec_ctx->pix_fmt;
        vs_ctx->codec->width = codec_ctx->width;
        vs_ctx->codec->height = codec_ctx->height;
        vs_ctx->codec->gop_size = codec_ctx->gop_size;
        vs_ctx->codec->field_order = codec_ctx->field_order;
        vs_ctx->codec->color_range = codec_ctx->color_range;
        vs_ctx->codec->color_primaries = codec_ctx->color_primaries;
        vs_ctx->codec->color_trc = codec_ctx->color_trc;
        vs_ctx->codec->colorspace = codec_ctx->colorspace;
        vs_ctx->codec->chroma_sample_location = codec_ctx->chroma_sample_location;
        vs_ctx->codec->sample_aspect_ratio = codec_ctx->sample_aspect_ratio;
        vs_ctx->codec->has_b_frames = codec_ctx->has_b_frames;
        break;
    case AVMEDIA_TYPE_AUDIO:
        vs_ctx->codec->sample_fmt = codec_ctx->sample_fmt;
        vs_ctx->codec->channel_layout = codec_ctx->channel_layout;
        vs_ctx->codec->channels = codec_ctx->channels;
        vs_ctx->codec->sample_rate = codec_ctx->sample_rate;
        vs_ctx->codec->block_align = codec_ctx->block_align;
        vs_ctx->codec->frame_size = codec_ctx->frame_size;
        vs_ctx->codec->initial_padding = codec_ctx->initial_padding;
        vs_ctx->codec->seek_preroll = codec_ctx->seek_preroll;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        vs_ctx->codec->width = codec_ctx->width;
        vs_ctx->codec->height = codec_ctx->height;
        break;
    default:
        break;
    }

    if (codec_ctx->extradata)
    {
        vs_ctx->codec->extradata_size = 0;
        av_free(vs_ctx->codec->extradata);

        vs_ctx->codec->extradata = (uint8_t *)av_mallocz(codec_ctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!vs_ctx->codec->extradata)
            return AVERROR(ENOMEM);
        
        memcpy(vs_ctx->codec->extradata, codec_ctx->extradata, codec_ctx->extradata_size);
        vs_ctx->codec->extradata_size = codec_ctx->extradata_size;

    }
#endif //FF_API_LAVF_AVCTX

    return 0;

#endif //#ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_stream_parameters_to_context(AVCodecContext *codec_ctx, const AVStream *vs_ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return -1;
#else //#ifndef HAVE_FFMPEG
    assert(vs_ctx != NULL && codec_ctx != NULL);

    /**/
    avcodec_parameters_to_context(codec_ctx, vs_ctx->codecpar);

    /*下面的也要复制，因为一些定制的ffmpeg未完成启用新的参数。*/
#if FF_API_LAVF_AVCTX
    codec_ctx->time_base = vs_ctx->codec->time_base;
    codec_ctx->framerate = vs_ctx->codec->framerate;
    codec_ctx->codec_type = vs_ctx->codec->codec_type;
    codec_ctx->codec_id = vs_ctx->codec->codec_id;
    codec_ctx->codec_tag = vs_ctx->codec->codec_tag;
    codec_ctx->bit_rate = vs_ctx->codec->bit_rate;
    codec_ctx->bits_per_coded_sample = vs_ctx->codec->bits_per_coded_sample;
    codec_ctx->bits_per_raw_sample = vs_ctx->codec->bits_per_raw_sample;
    codec_ctx->profile = vs_ctx->codec->profile;
    codec_ctx->level = vs_ctx->codec->level;
    codec_ctx->flags = vs_ctx->codec->flags;

    switch (vs_ctx->codec->codec_type)
    {
    case AVMEDIA_TYPE_VIDEO:
        codec_ctx->pix_fmt = vs_ctx->codec->pix_fmt;
        codec_ctx->width = vs_ctx->codec->width;
        codec_ctx->height = vs_ctx->codec->height;
        codec_ctx->field_order = vs_ctx->codec->field_order;
        codec_ctx->color_range = vs_ctx->codec->color_range;
        codec_ctx->color_primaries = vs_ctx->codec->color_primaries;
        codec_ctx->color_trc = vs_ctx->codec->color_trc;
        codec_ctx->colorspace = vs_ctx->codec->colorspace;
        codec_ctx->chroma_sample_location = vs_ctx->codec->chroma_sample_location;
        codec_ctx->sample_aspect_ratio = vs_ctx->codec->sample_aspect_ratio;
        codec_ctx->has_b_frames = vs_ctx->codec->has_b_frames;
        break;
    case AVMEDIA_TYPE_AUDIO:
        codec_ctx->sample_fmt = vs_ctx->codec->sample_fmt;
        codec_ctx->channel_layout = vs_ctx->codec->channel_layout;
        codec_ctx->channels = vs_ctx->codec->channels;
        codec_ctx->sample_rate = vs_ctx->codec->sample_rate;
        codec_ctx->block_align = vs_ctx->codec->block_align;
        codec_ctx->frame_size = vs_ctx->codec->frame_size;
        codec_ctx->delay = codec_ctx->initial_padding = vs_ctx->codec->initial_padding;
        codec_ctx->seek_preroll = vs_ctx->codec->seek_preroll;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        codec_ctx->width = vs_ctx->codec->width;
        codec_ctx->height = vs_ctx->codec->height;
        break;
    default:
        break;
    }

    if (vs_ctx->codec->extradata)
    {
        codec_ctx->extradata_size = 0;
        av_free(codec_ctx->extradata);

        codec_ctx->extradata = (uint8_t *)av_mallocz(vs_ctx->codec->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!codec_ctx->extradata)
            return AVERROR(ENOMEM);

        memcpy(codec_ctx->extradata, vs_ctx->codec->extradata, vs_ctx->codec->extradata_size);
        codec_ctx->extradata_size = vs_ctx->codec->extradata_size;
    }
#endif //FF_API_LAVF_AVCTX

    return 0;

#endif //#ifndef HAVE_FFMPEG
}