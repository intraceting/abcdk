/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/util.h"

void abcdk_ffmpeg_library_deinit()
{
    avformat_network_deinit();
}

void abcdk_ffmpeg_library_init()
{
    avformat_network_init();
    avdevice_register_all();
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    avcodec_register_all();
#endif //#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
}

void abcdk_ffmpeg_io_free(AVIOContext **ctx)
{
    AVIOContext *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->buffer)
        av_free(ctx_p->buffer);

#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
    avio_context_free(&ctx_p);
#else
    av_free(ctx_p);
#endif
}

AVIOContext *abcdk_ffmpeg_io_alloc(int buf_blocks, int write_flag)
{
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
}

void abcdk_ffmpeg_media_dump(AVFormatContext *ctx,int output)
{
    if (!ctx)
        return;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    av_dump_format(ctx, 0, ctx->filename, output);
#else
    av_dump_format(ctx, 0, ctx->url, output);
#endif
}

void abcdk_ffmpeg_media_option_dump(AVFormatContext *ctx)
{
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
}

void abcdk_ffmpeg_media_free(AVFormatContext **ctx)
{
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
}

int abcdk_avformat_media_open(AVFormatContext **ctx, int writer, const char *fmt, const char *url, AVIOContext *vio_ctx,
                                    AVIOInterruptCB *inter_ctx, AVDictionary **options)
{
    AVInputFormat *fmt_ctx = NULL;
    AVFormatContext *media_ctx = NULL;
    int chk;

    assert(ctx != NULL);
    assert(url != NULL || (fmt != NULL && vio_ctx != NULL));

    //free old context.
    abcdk_ffmpeg_media_free(ctx);

    media_ctx = avformat_alloc_context();
    if (!media_ctx)
        return AVERROR(ENOMEM);

    /*
     * 1: 如果不知道下面标志如何使用，一定不要附加这个标志。
     * 2: 如果附加此标志，会造成数据流开头的数据包丢失(N个)。
     * 3: 如果未附加此标志，网络流会产生不确定的延时(N毫秒~N秒)。
    */
    //media_ctx->flags |= AVFMT_FLAG_NOBUFFER;

    if (inter_ctx)
        media_ctx->interrupt_callback = *inter_ctx;

    if (vio_ctx)
    {
        media_ctx->pb = vio_ctx;
        media_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
    }

    fmt_ctx = (AVInputFormat *)av_find_input_format(fmt);
    chk = avformat_open_input(&media_ctx, url, fmt_ctx, options);

    if (chk != 0)
        abcdk_ffmpeg_media_free(&media_ctx);

    //copy, may be is NULL.
    *ctx = media_ctx;

    return chk;
}

int abcdk_avformat_media_open_output(AVFormatContext **ctx, const char *fmt, const char *url, AVIOContext *vio_ctx,
                                     AVIOInterruptCB *inter_ctx, AVDictionary **options)
{
    AVInputFormat *fmt_ctx = NULL;
    AVFormatContext *media_ctx = NULL;
    int chk;

    assert(ctx != NULL);
    assert(url != NULL || (fmt != NULL && vio_ctx != NULL));

    //free old context.
    abcdk_ffmpeg_media_free(ctx);

    if (abcdk_strncmp(url, "rtsp://", 7, 0) == 0)
        ctx->oformat = av_guess_format("rtsp", NULL, NULL);
    else if (abcdk_strncmp(url, "rtsps://", 8, 0) == 0)
        ctx->oformat = av_guess_format("rtsp", NULL, NULL);
    else if (abcdk_strncmp(url, "rtmp://", 7, 0) == 0)
        ctx->oformat = av_guess_format("flv", NULL, NULL);
    else if (abcdk_strncmp(url, "rtmps://", 8, 0) == 0)
        ctx->oformat = av_guess_format("flv", NULL, NULL);

    media_ctx = avformat_alloc_context();
    if (!media_ctx)
        return AVERROR(ENOMEM);

    if (url)
    {
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
        strncpy(ctx->filename, url, sizeof(ctx->filename));
#else
        ctx->url = av_strdup(url);
#endif
    }

    //
    ctx->oformat = av_guess_format(fmt, url, NULL);

    if (!ctx->oformat)
        goto final_error;
    
#if 0
    av_dict_set(&ctx->metadata, "service", ABCDK, 0);
    av_dict_set(&ctx->metadata, "service_name", ABCDK, 0);
    av_dict_set(&ctx->metadata, "service_provider", ABCDK, 0);
    av_dict_set(&ctx->metadata, "artist", ABCDK, 0);
#endif

    if (inter_ctx)
        ctx->interrupt_callback = *inter_ctx;

    if (io)
    {
        ctx->pb = io;
        ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
        ctx->flags |= AVFMT_FLAG_FLUSH_PACKETS;
    }

    return ctx;

final_error:

    abcdk_avformat_free(&ctx);

    return NULL;
}

void abcdk_ffmpeg_codec_option_dump(AVCodec *ctx)
{
    if (!ctx)
        return;

    if (ctx->priv_class)
        av_opt_show2((void *)&ctx->priv_class, NULL, -1, 0);
    else
        av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->long_name ? ctx->long_name : ctx->name));
}

void abcdk_ffmpeg_codec_free(AVCodecContext **ctx)
{
    AVCodecContext *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p)
        avcodec_close(ctx_p);

    avcodec_free_context(&ctx_p);
}
