/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/editor.h"

struct _abcdk_ffmpeg_editor
{
    int writer;
    abcdk_editor_param_t param;
    
    AVDictionary *option_ctx;
    AVFormatContext *media_ctx;
    AVIOContext *vio_ctx;

    uint64_t read_last_time;
    int write_header_ok;

};// abcdk_ffmpeg_editor_t;

void abcdk_ffmpeg_editor_free(abcdk_ffmpeg_editor_t **ctx)
{
    abcdk_ffmpeg_editor_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_heap_freep(&ctx_p->param.fmt);
    abcdk_heap_freep(&ctx_p->param.url);

    av_dict_free(&ctx_p->option_ctx);
    abcdk_ffmpeg_media_free(&ctx_p->media_ctx);
    abcdk_ffmpeg_io_free(&ctx_p->vio_ctx);
}

abcdk_ffmpeg_editor_t *abcdk_ffmpeg_editor_alloc(int writer)
{
    abcdk_ffmpeg_editor_t *ctx;

    ctx = new abcdk_ffmpeg_editor_t;
    if(!ctx)
        return NULL;
    
    ctx->writer = writer;

    return ctx;
}

static int _abcdk_ffmpeg_editor_interrupt_cb(void *args)
{
    abcdk_ffmpeg_editor_t *ctx = (abcdk_ffmpeg_editor_t *)args;
    uint64_t check_time = abcdk_time_systime(0);

    //允许未启用.
    if (ctx->param.timeout <= 0)
        return 0;

    //如果是作者并且已经连接成功, 则忽略超时检测.
    if (ctx->writer && ctx->write_header_ok)
        return 0;

    //超时检测, 未超时返回0, 否则返回-1.
    if ((check_time - ctx->read_last_time) < ctx->param.timeout)
        return 0;
    
    return -1;
}

static int _abcdk_avformat_media_init(abcdk_ffmpeg_editor_t *ctx)
{
    AVInputFormat *fmt_ctx = NULL;
    AVFormatContext *media_ctx = NULL;
    int chk;

    //free old context.
    abcdk_ffmpeg_media_free(&ctx->media_ctx);

    ctx->media_ctx = avformat_alloc_context();
    if (!ctx->media_ctx)
        return AVERROR(ENOMEM);


    ctx->media_ctx->interrupt_callback.callback = _abcdk_ffmpeg_editor_interrupt_cb;
    ctx->media_ctx->interrupt_callback.opaque = ctx;

    if (ctx->vio_ctx)
    {
        ctx->media_ctx->pb = ctx->vio_ctx;
        ctx->media_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
    }

    if (ctx->writer)
    {
#if 0
        av_dict_set(&ctx->media_ctx->metadata, "service", ABCDK, 0);
        av_dict_set(&ctx->media_ctx->metadata, "service_name", ABCDK, 0);
        av_dict_set(&ctx->media_ctx->metadata, "service_provider", ABCDK, 0);
        av_dict_set(&ctx->media_ctx->metadata, "artist", ABCDK, 0);
#endif

        if(ctx->param.write_nodelay)
            ctx->flags |= AVFMT_FLAG_FLUSH_PACKETS;

        ctx->media_ctx->oformat = av_guess_format(ctx->param.fmt, ctx->param.url, NULL);
    }
    else
    {
        // 更新时间, 否则重复打开时会有超时发生.
        ctx->read_last_time = abcdk_time_systime(0);

        /*
         * 1: 如果不知道下面标志如何使用，一定不要附加这个标志。
         * 2: 如果附加此标志，会造成数据流开头的数据包丢失(N个)。
         * 3: 如果未附加此标志，网络流会产生不确定的延时(N毫秒~N秒)。
         */
        // media_ctx->flags |= AVFMT_FLAG_NOBUFFER;

        fmt_ctx = (AVInputFormat *)av_find_input_format(ctx->param.fmt);
        chk = avformat_open_input(&ctx->media_ctx, ctx->param.url, fmt_ctx, ctx->option_ctx);
        if (chk != 0)
        {
            abcdk_ffmpeg_media_free(&ctx->media_ctx);
            return chk;
        }

        chk = avformat_find_stream_info(ctx->media_ctx, NULL);
        if (chk < 0)
        {
            abcdk_ffmpeg_media_free(&ctx->media_ctx);
            return chk;
        }
    }

    abcdk_ffmpeg_media_dump(ctx->media_ctx, ctx->writer);
    // abcdk_ffmpeg_media_option_dump(ctx->media_ctx);

    return 0;
}



int abcdk_ffmpeg_editor_open(abcdk_ffmpeg_editor_t *ctx, abcdk_editor_param_t *param)
{
    int chk;

    assert(ctx != NULL && param != NULL);

    ctx->param = *param;

    //覆盖外部指针.
    ctx->param.fmt = abcdk_strdup_safe(param->fmt);
    ctx->param.url = abcdk_strdup_safe(param->url);

    if (ctx->param.url)
    {
        if (ctx->writer)
        {
            if (!ctx->param.fmt)
            {
                //下面的网络协议, 如果未指定格式, 需要内部自行判定. 
                if (abcdk_strncmp(url, "rtsp://", 7, 0) == 0)
                    ctx->param.fmt = abcdk_strdup_safe("rtsp");
                else if (abcdk_strncmp(url, "rtsps://", 8, 0) == 0)
                    ctx->param.fmt = abcdk_strdup_safe("rtsp");
                else if (abcdk_strncmp(url, "rtmp://", 7, 0) == 0)
                    ctx->param.fmt = abcdk_strdup_safe("flv");
                else if (abcdk_strncmp(url, "rtmps://", 8, 0) == 0)
                    ctx->param.fmt = abcdk_strdup_safe("flv");
            }
        }
        else
        {
            av_dict_set_int(&ctx->dict_ctx, "stimeout", ctx->param.timeout * 1000000, 0);   // rtsp
            av_dict_set_int(&ctx->dict_ctx, "rw_timeout", ctx->param.timeout * 1000000, 0); // rtmp

            /* RTSP默认走TCP，可以减少丢包。*/
            if (strncmp(ctx->param.url, "rtsp://", 7) == 0 ||
                strncmp(ctx->param.url, "rtsps://", 8) == 0)
            {
                av_dict_set(&ctx->dict_ctx, "rtsp_transport", "tcp", 0);
                chk = _abcdk_avformat_media_init(ctx->media_ctx);
                if (chk < 0)
                {
                    av_dict_set(&ctx->dict_ctx, "rtsp_transport", "udp", 0);
                    chk = _abcdk_avformat_media_init(ctx->media_ctx);
                    if (chk < 0)
                        return chk;
                }
            }
            else
            {
                chk = _abcdk_avformat_media_init(ctx->media_ctx);
                if (chk < 0)
                    return chk;
            }
        }
    }
    else if(ctx->param.vio.read_cb || ctx->param.vio.write_cb)
    {
        ctx->vio_ctx = abcdk_ffmpeg_io_alloc(8,ctx->writer);
        if(!ctx->vio_ctx)
            return AVERROR(ENOMEM);

        ctx->vio_ctx->opaque = ctx->param.vio.opaque;
        ctx->vio_ctx->read_packet = ctx->param.vio.read_cb;
        ctx->vio_ctx->write_packet = ctx->param.vio.write_cb;
  
        chk = _abcdk_avformat_media_init(ctx->media_ctx);
        if (chk < 0)
            return chk;

        ctx->vio_ctx = NULL;//It has been hosted and can no longer be used.
    }
    else
    {
        return AVERROR(EINVAL);
    }


    return 0;
}
