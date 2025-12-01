/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/util.h"

void abcdk_ffmpeg_deinit()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else  // #ifndef HAVE_FFMPEG
    avformat_network_deinit();
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_init()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else // #ifndef HAVE_FFMPEG
    avformat_network_init();
    avdevice_register_all();
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    avcodec_register_all();
#endif // #if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)

#endif // #ifndef HAVE_FFMPEG
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

#endif // #ifdef HAVE_FFMPEG

void abcdk_ffmpeg_log_redirect()
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else  // #ifndef HAVE_FFMPEG
    av_log_set_callback(_abcdk_ffmpeg_log_callback);
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_io_free(AVIOContext **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else // #ifndef HAVE_FFMPEG
    AVIOContext *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->buffer)
        av_free(ctx_p->buffer);

#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
    avio_context_free(&ctx_p);
#else  // #if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
    av_free(ctx_p);
#endif // #if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)

#endif // #ifndef HAVE_FFMPEG
}

AVIOContext *abcdk_ffmpeg_io_alloc(int buf_blocks, int write_flag)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return NULL;
#else  // #ifndef HAVE_FFMPEG
    int buf_size = 8 * 4096; /* 4k bytes 的倍数. */
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
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_media_dump(AVFormatContext *ctx, int output)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else // #ifndef HAVE_FFMPEG
    if (!ctx)
        return;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    av_dump_format(ctx, 0, ctx->filename, output);
#else  // #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    av_dump_format(ctx, 0, ctx->url, output);
#endif // #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)

#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_media_option_dump(AVFormatContext *ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else  // #ifndef HAVE_FFMPEG
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
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_media_free(AVFormatContext **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else  // #ifndef HAVE_FFMPEG
    AVFormatContext *ctx_p = NULL;
    AVIOContext *pb = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*
     * IO对象在以下两个条件下，需要单独释放.
     *
     * 1：自定义环境.
     * 2：输出对象.
     */
    if (ctx_p->flags & AVFMT_FLAG_CUSTOM_IO)
        abcdk_ffmpeg_io_free(&ctx_p->pb);
    else if (ctx_p->oformat && !(ctx_p->oformat->flags & AVFMT_NOFILE))
        avio_closep(&ctx_p->pb);

    if (ctx_p->iformat)
        avformat_close_input(&ctx_p);
    else
        avformat_free_context(ctx_p);
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_codec_option_dump(AVCodec *ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else  // #ifndef HAVE_FFMPEG
    if (!ctx)
        return;

    if (ctx->priv_class)
        av_opt_show2((void *)&ctx->priv_class, NULL, -1, 0);
    else
        av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->long_name ? ctx->long_name : ctx->name));
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_codec_free(AVCodecContext **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else  // #ifndef HAVE_FFMPEG
    AVCodecContext *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p)
        avcodec_close(ctx_p);

    avcodec_free_context(&ctx_p);
#endif // #ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_q2d(AVRational *r, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return 0.0;
#else  // #ifndef HAVE_FFMPEG
    double d = (r->num == 0 || r->den == 0 ? 0. : av_q2d(*r));

    return scale * d;
#endif // #ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_stream_timebase_q2d(AVStream *vs_ctx, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return 0.0;
#else  // #ifndef HAVE_FFMPEG
    assert(vs_ctx != NULL);

    return abcdk_ffmpeg_q2d(&vs_ctx->time_base, scale);
#endif // #ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_stream_duration(AVStream *vs_ctx, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return 0.0;
#else  // #ifndef HAVE_FFMPEG
    double sec = 0.0;

    assert(vs_ctx != NULL);

    if (vs_ctx->duration > 0)
        sec = (double)vs_ctx->duration * abcdk_ffmpeg_q2d(&vs_ctx->time_base, scale);

    return sec;
#endif // #ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_stream_time2rate(AVStream *vs_ctx, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return 0.000001;
#else // #ifndef HAVE_FFMPEG
    double rate = 0.000001, base_rate = 0.000025;

    assert(vs_ctx != NULL);

#if LIBAVCODEC_BUILD >= AV_VERSION_INT(54, 1, 0)
    rate = abcdk_ffmpeg_q2d(&vs_ctx->avg_frame_rate, scale);
#else  // #if LIBAVCODEC_BUILD >= AV_VERSION_INT(54, 1, 0)
    rate = abcdk_ffmpeg_q2d(&vs_ctx->r_frame_rate, scale);
#endif // #if LIBAVCODEC_BUILD >= AV_VERSION_INT(54, 1, 0)

    if (rate < base_rate)
#if FF_API_LAVF_AVCTX
        rate = 1.0 / abcdk_ffmpeg_q2d(&vs_ctx->codec->time_base, scale);
#else  // FF_API_LAVF_AVCTX
        rate = 1.0 / abcdk_ffmpeg_q2d(&vs_ctx->time_base, scale);
#endif // FF_API_LAVF_AVCTX

    return ABCDK_CLAMP(rate, (double)base_rate, (double)999999999.0);
#endif // #ifndef HAVE_FFMPEG
}

double abcdk_ffmpeg_stream_ts2sec(AVStream *vs_ctx, int64_t ts, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -0.000001;
#else  // #ifndef HAVE_FFMPEG
    double sec;

    assert(vs_ctx != NULL);

    sec = (double)(ts - vs_ctx->start_time) * abcdk_ffmpeg_q2d(&vs_ctx->time_base, 1.0 / scale);

    return sec;
#endif // #ifndef HAVE_FFMPEG
}

int64_t abcdk_ffmpeg_stream_ts2num(AVStream *vs_ctx, int64_t ts, double scale)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    int64_t num;
    double sec;

    assert(vs_ctx != NULL);

    sec = abcdk_ffmpeg_stream_ts2sec(vs_ctx, ts, scale);
    num = (int64_t)(abcdk_ffmpeg_stream_time2rate(vs_ctx, scale) * sec + 0.5);

    return num;
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_stream_fix_bitrate(AVStream *vs_ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else // #ifndef HAVE_FFMPEG
    int64_t bit_rate;
    double video_fps;

    assert(vs_ctx != NULL);

    if (vs_ctx->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        video_fps = abcdk_ffmpeg_stream_time2rate(vs_ctx, 1.0);

        if (vs_ctx->codecpar->codec_id == AV_CODEC_ID_H264)
            bit_rate = vs_ctx->codecpar->width * vs_ctx->codecpar->height * video_fps * 0.1; // 经验估算.
        else if (vs_ctx->codecpar->codec_id == AV_CODEC_ID_H265)
            bit_rate = vs_ctx->codecpar->width * vs_ctx->codecpar->height * video_fps * 0.6; // 经验估算.
        else
            bit_rate = vs_ctx->codecpar->width * vs_ctx->codecpar->height * video_fps * 0.3; // fix me.
    }
    else if (vs_ctx->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
    {
        bit_rate = vs_ctx->codecpar->sample_rate * vs_ctx->codecpar->channels * 0.12; // 经验估算.
    }
    else
    {
        return;
    }

#if FF_API_LAVF_AVCTX
    if (vs_ctx->codecpar->bit_rate <= 0 || vs_ctx->codec->bit_rate <= 0)
#else  // #if FF_API_LAVF_AVCTX
    if (vs_ctx->codecpar->bit_rate <= 0)
#endif // #if FF_API_LAVF_AVCTX
    {
        vs_ctx->codecpar->bit_rate = bit_rate;
#if FF_API_LAVF_AVCTX
        vs_ctx->codec->bit_rate = bit_rate;
        vs_ctx->codec->bit_rate_tolerance = bit_rate;
#endif // #if FF_API_LAVF_AVCTX
    }
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_pixfmt_get_bit(AVPixelFormat pixfmt, int have_pad)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    const AVPixFmtDescriptor *desc;

    assert(pixfmt > AV_PIX_FMT_NONE);

    desc = av_pix_fmt_desc_get(pixfmt);
    if (desc)
        return (have_pad ? av_get_padded_bits_per_pixel(desc) : av_get_bits_per_pixel(desc));

    return -1;
#endif // #ifndef HAVE_FFMPEG
}

const char *abcdk_ffmpeg_pixfmt_get_name(AVPixelFormat pixfmt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return NULL;
#else  // #ifndef HAVE_FFMPEG
    const AVPixFmtDescriptor *desc;

    assert(pixfmt > AV_PIX_FMT_NONE);

    desc = av_pix_fmt_desc_get(pixfmt);
    if (desc)
        return av_get_pix_fmt_name(pixfmt);

    return NULL;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_pixfmt_get_channel(AVPixelFormat pixfmt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    const AVPixFmtDescriptor *desc;

    assert(pixfmt > AV_PIX_FMT_NONE);

    desc = av_pix_fmt_desc_get(pixfmt);
    if (desc)
        return desc->nb_components;

    return 0;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_image_fill_height(int heights[4], int height, AVPixelFormat pixfmt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    const AVPixFmtDescriptor *desc;
    int planes_nb;
    int h;

    assert(heights != NULL && height > 0 && pixfmt > AV_PIX_FMT_NONE);

    desc = av_pix_fmt_desc_get(pixfmt);
    if (!desc)
        return -1;

    planes_nb = 0;
    for (int i = 0; i < desc->nb_components; i++)
        planes_nb = FFMAX(planes_nb, desc->comp[i].plane + 1);

    if (planes_nb <= 4)
    {
        for (int i = 0; i < planes_nb; i++)
        {
            h = height;
            if (i == 1 || i == 2)
            {
                h = FF_CEIL_RSHIFT(height, desc->log2_chroma_h);
            }

            heights[i] = h;
        }
    }

    return planes_nb;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_image_fill_stride(int stride[4], int width, AVPixelFormat pixfmt, int align)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    int planes_nb;

    assert(stride != NULL && width > 0 && pixfmt > AV_PIX_FMT_NONE);

    if (av_image_fill_linesizes(stride, pixfmt, width) < 0)
        return -1;

    planes_nb = 0;
    for (int i = 0; i < 4; i++)
    {
        if (stride[i] <= 0)
            continue;

        stride[i] = abcdk_align(stride[i], align);
        planes_nb += 1;
    }

    return planes_nb;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_image_fill_pointer(uint8_t *data[4], const int stride[4], int height, AVPixelFormat pixfmt, void *buffer)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    int size;

    assert(data != NULL && stride != NULL && height > 0 && pixfmt > AV_PIX_FMT_NONE);

    size = av_image_fill_pointers(data, pixfmt, height, (uint8_t *)buffer, stride);

    /*如果只是需要的内存大小, 则清空无效指针.*/
    if (!buffer)
        data[0] = data[1] = data[2] = data[3] = NULL;

    return size;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_image_get_size(const int stride[4], int height, AVPixelFormat pixfmt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    uint8_t *data[4] = {0};

    return abcdk_ffmpeg_image_fill_pointer(data, stride, height, pixfmt, NULL);
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_image_get_size2(int width, int height, AVPixelFormat pixfmt, int align)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else  // #ifndef HAVE_FFMPEG
    int stride[4] = {0};
    int chk;

    chk = abcdk_ffmpeg_image_fill_stride(stride, width, pixfmt, align);
    if (chk <= 0)
        return chk;

    return abcdk_ffmpeg_image_get_size(stride, height, pixfmt);
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_image_copy(uint8_t *dst_data[4], int dst_stride[4],
                             const uint8_t *src_data[4], const int src_stride[4],
                             int width, int height, AVPixelFormat pixfmt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else  // #ifndef HAVE_FFMPEG
    assert(dst_data != NULL && dst_stride != NULL);
    assert(src_data != NULL && src_stride != NULL);
    assert(width > 0 && height > 0 && pixfmt > AV_PIX_FMT_NONE);

    av_image_copy(dst_data, dst_stride, src_data, src_stride, pixfmt, width, height);
#endif // #ifndef HAVE_FFMPEG
}

void abcdk_ffmpeg_image_copy2(AVFrame *dst, const AVFrame *src)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含FFMPEG工具."));
    return;
#else  // #ifndef HAVE_FFMPEG
    assert(dst != NULL && src != NULL);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->format == src->format);

    abcdk_ffmpeg_image_copy(dst->data, dst->linesize, (const uint8_t **)src->data, src->linesize,
                            src->width, src->height, (AVPixelFormat)src->format);
#endif // #ifndef HAVE_FFMPEG
}