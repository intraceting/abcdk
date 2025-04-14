/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/rtsp/relay.h"

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

/**简单的RTSP中继服务。*/
struct _abcdk_rtsp_relay
{
    /**退出标志。0 运行，!0 退出。*/
    volatile int exit_flag;

    /**工作线程。 */
    abcdk_thread_t worker_thread;

    /**FF配置。 */
    abcdk_ffeditor_config_t ff_cfg;

    /**FF环境。 */
    abcdk_ffeditor_t *ff_ctx;

    /**FF环境。 */
    abcdk_ffsocket_t *ff_sock;

    /**流索引映射表。 */
    int index_s2d[16];

    /**服务指针。 */
    abcdk_rtsp_server_t *server_ctx_p;

    /**媒体名字。*/
    char *media_name;

    /**源地址。*/
    char *src_url;

    /**源格式。*/
    char *src_fmt;

    /**源倍速。*/
    float src_xspeed;

    /**源超时(秒)。*/
    int src_timeout;

    /**源重试间隔(秒)。*/
    int src_retry;

}; // abcdk_rtsp_relay_t;

static void _abcdk_rtsp_relay_worker_thread_stop(abcdk_rtsp_relay_t *ctx)
{
    ctx->exit_flag = 1;
    abcdk_thread_join(&ctx->worker_thread);
}

static void *_abcdk_rtsp_relay_worker_thread_routine(void *opaque);

static int _abcdk_rtsp_relay_worker_thread_start(abcdk_rtsp_relay_t *ctx)
{
    int chk;

    ctx->exit_flag = 0;
    ctx->worker_thread.routine = _abcdk_rtsp_relay_worker_thread_routine;
    ctx->worker_thread.opaque = ctx;

    chk = abcdk_thread_create(&ctx->worker_thread, 1);
    if (chk != 0)
        return -1;

    return 0;
}

void abcdk_rtsp_relay_destroy(abcdk_rtsp_relay_t **ctx)
{
    abcdk_rtsp_relay_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*工作线程停止。*/
    _abcdk_rtsp_relay_worker_thread_stop(ctx_p);

    abcdk_heap_free(ctx_p->media_name);
    abcdk_heap_free(ctx_p->src_url);
    abcdk_heap_free(ctx_p->src_fmt);

    abcdk_heap_free(ctx_p);
}

abcdk_rtsp_relay_t *abcdk_rtsp_relay_create(abcdk_rtsp_server_t *server_ctx, const char *media_name, const char *src_url, const char *src_fmt, float src_xspeed, int src_timeout, int src_retry)
{
    abcdk_rtsp_relay_t *ctx;
    int chk;

    assert(server_ctx != NULL && media_name != NULL && src_url != NULL && src_timeout > 0 && src_retry > 0);

    ctx = (abcdk_rtsp_relay_t *)abcdk_heap_alloc(sizeof(abcdk_rtsp_relay_t));
    if (!ctx)
        return NULL;

    ctx->server_ctx_p = server_ctx;
    ctx->media_name = abcdk_heap_clone(media_name, strlen(media_name));
    ctx->src_url = abcdk_heap_clone(src_url, strlen(src_url));
    ctx->src_fmt = (src_fmt ? abcdk_heap_clone(src_fmt, strlen(src_fmt)) : NULL);
    ctx->src_xspeed = src_xspeed;
    ctx->src_timeout = src_timeout;
    ctx->src_retry = src_retry;

    chk = _abcdk_rtsp_relay_worker_thread_start(ctx);
    if (chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_rtsp_relay_destroy(&ctx);
    return NULL;
}

static int _abcdk_rtsp_realy_create_media(abcdk_rtsp_relay_t *ctx)
{
    abcdk_object_t *extdata = NULL;
    int bitrate = 0;
    int cache = 0;
    AVStream *src_vs_p = NULL;

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    AVCodecContext *codecpar = NULL;
#else
    AVCodecParameters *codecpar = NULL;
#endif

    int chk;

    /*创建新的。*/
    chk = abcdk_rtsp_server_create_media(ctx->server_ctx_p, ctx->media_name, NULL, NULL);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING, TT("媒体(%s)已存在，或其它错误。"), ctx->media_name);
        return -1;
    }

    for (int i = 0; i < abcdk_ffeditor_streams(ctx->ff_ctx); i++)
    {
        abcdk_object_unref(&extdata);

        src_vs_p = abcdk_ffeditor_streamptr(ctx->ff_ctx, i);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
        codecpar = src_vs_p->codec;
#else
        codecpar = src_vs_p->codecpar;
#endif

        /*获取流的帧率作为缓存长度。*/
        cache = abcdk_ffeditor_fps(ctx->ff_ctx, i);
        /*码率转换(bps -> kbps)。*/
        bitrate = (codecpar->bit_rate > 0 ? codecpar->bit_rate / 1000 : 0);

        if (codecpar->codec_id == AV_CODEC_ID_HEVC)
        {
            extdata = abcdk_object_copyfrom(codecpar->extradata, codecpar->extradata_size);

            ctx->index_s2d[i] = abcdk_rtsp_server_add_stream(ctx->server_ctx_p, ctx->media_name, ABCDK_RTSP_CODEC_H265, extdata, ABCDK_CLAMP(bitrate, 3000, 50000), ABCDK_CLAMP(cache, 25, 100));
        }
        else if (codecpar->codec_id == AV_CODEC_ID_H264)
        {
            extdata = abcdk_object_copyfrom(codecpar->extradata, codecpar->extradata_size);

            ctx->index_s2d[i] = abcdk_rtsp_server_add_stream(ctx->server_ctx_p, ctx->media_name, ABCDK_RTSP_CODEC_H264, extdata, ABCDK_CLAMP(bitrate, 3000, 50000), ABCDK_CLAMP(cache, 25, 100));
        }
        else if (codecpar->codec_id == AV_CODEC_ID_AAC)
        {
            extdata = abcdk_object_copyfrom(codecpar->extradata, codecpar->extradata_size);

            ctx->index_s2d[i] = abcdk_rtsp_server_add_stream(ctx->server_ctx_p, ctx->media_name, ABCDK_RTSP_CODEC_AAC, extdata, ABCDK_CLAMP(bitrate, 96, 512), ABCDK_CLAMP(cache, 5, 10));
        }
        else if (codecpar->codec_id == AV_CODEC_ID_PCM_MULAW)
        {
            extdata = abcdk_object_alloc3(sizeof(int), 2); //[0] = channels,[1]=sample_rate
            ABCDK_PTR2I32(extdata->pptrs[0], 0) = codecpar->channels;
            ABCDK_PTR2I32(extdata->pptrs[1], 0) = codecpar->sample_rate;

            ctx->index_s2d[i] = abcdk_rtsp_server_add_stream(ctx->server_ctx_p, ctx->media_name, ABCDK_RTSP_CODEC_G711U, extdata, ABCDK_CLAMP(bitrate, 96, 512), ABCDK_CLAMP(cache, 5, 10));
        }
        else if (codecpar->codec_id == AV_CODEC_ID_PCM_ALAW)
        {
            extdata = abcdk_object_alloc3(sizeof(int), 2); //[0] = channels,[1]=sample_rate
            ABCDK_PTR2I32(extdata->pptrs[0], 0) = codecpar->channels;
            ABCDK_PTR2I32(extdata->pptrs[1], 0) = codecpar->sample_rate;

            ctx->index_s2d[i] = abcdk_rtsp_server_add_stream(ctx->server_ctx_p, ctx->media_name, ABCDK_RTSP_CODEC_G711A, extdata, ABCDK_CLAMP(bitrate, 96, 512), ABCDK_CLAMP(cache, 5, 10));
        }
        else
        {
            ctx->index_s2d[i] = -1;

            abcdk_trace_printf(LOG_WARNING, TT("媒体(%s)不支持编码格式(FF-CODEC-ID=%d)，忽略。"), ctx->media_name, codecpar->codec_id);
        }
    }

    abcdk_object_unref(&extdata); // free.

    chk = abcdk_rtsp_server_play_media(ctx->server_ctx_p, ctx->media_name);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING, TT("媒体(%s)不存在，或没有需要播放的流。"), ctx->media_name);
        return -1;
    }

    return 0;
}

static void _abcdk_rtsp_realy_forward_media(abcdk_rtsp_relay_t *ctx)
{
    AVPacket pkt;
    int src_idx, dst_idx;
    double pts_sec, dur_sec;

    av_init_packet(&pkt);

    while (!ctx->exit_flag)
    {
        src_idx = abcdk_ffeditor_read_packet(ctx->ff_ctx, &pkt, -1);
        if (src_idx < 0)
            break;

        assert(pkt.stream_index == src_idx);

        dst_idx = ctx->index_s2d[src_idx];

        /*跳过不支持的流。*/
        if (dst_idx < 0)
            continue;

        pts_sec = abcdk_ffeditor_ts2sec(ctx->ff_ctx, src_idx, pkt.pts);                     // 秒。
        dur_sec = (double)pkt.duration * abcdk_ffeditor_timebase_q2d(ctx->ff_ctx, src_idx); // 秒。

        abcdk_rtsp_server_play_stream(ctx->server_ctx_p, ctx->media_name, dst_idx, pkt.data, pkt.size, pts_sec * 1000000, dur_sec * 1000000); // 转微秒。
    }

    av_packet_unref(&pkt);
}

static void _abcdk_rtsp_relay_process(abcdk_rtsp_relay_t *ctx)
{
    int retry_count = 0;
    int chk;


RETRY:

    if (ctx->exit_flag)
        goto END;

    abcdk_ffeditor_destroy(&ctx->ff_ctx);
    abcdk_ffsocket_destroy(&ctx->ff_sock);

    /*第一次连接时不需要休息。*/
    if (retry_count++ > 0)
    {
        abcdk_trace_printf(LOG_WARNING, TT("源(%s)已关闭或到末尾，%d秒后重连。"), ctx->src_url, ctx->src_retry);
        usleep(ctx->src_retry * 1000000);
    }

    abcdk_trace_printf(LOG_INFO, TT("打开源(%s)..."), ctx->src_url);

    ctx->ff_cfg.url = ctx->src_url;
    ctx->ff_cfg.fmt = ctx->src_fmt;
    ctx->ff_cfg.timeout = ctx->src_timeout;
    ctx->ff_cfg.read_speed = ctx->src_xspeed;
    ctx->ff_cfg.read_delay_max = 3.0;
    ctx->ff_cfg.bit_stream_filter = 1;

    ctx->ff_ctx = abcdk_ffeditor_open(&ctx->ff_cfg);
    if (!ctx->ff_ctx)
        goto RETRY;

    /*删除旧的。*/
    abcdk_rtsp_server_remove_media(ctx->server_ctx_p, ctx->media_name);

    /*创建新的。*/
    chk = _abcdk_rtsp_realy_create_media(ctx);
    if (chk != 0)
        goto RETRY;

    /*拉流转发。*/
    _abcdk_rtsp_realy_forward_media(ctx);

    /**/
    goto RETRY;

END:

    /*删除旧的。*/
    abcdk_rtsp_server_remove_media(ctx->server_ctx_p, ctx->media_name);

    abcdk_ffeditor_destroy(&ctx->ff_ctx);
    abcdk_ffsocket_destroy(&ctx->ff_sock);

    abcdk_trace_printf(LOG_INFO, TT("源(%s)关闭，退出。"), ctx->src_url);

    return;
}

static void *_abcdk_rtsp_relay_worker_thread_routine(void *opaque)
{
    abcdk_rtsp_relay_t *ctx = (abcdk_rtsp_relay_t *)opaque;

    /*设置线程名字，日志记录会用到。*/
    abcdk_thread_setname(0, "%x", abcdk_sequence_num());

    _abcdk_rtsp_relay_process(ctx);

    return NULL;
}

#else // #if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

void abcdk_rtsp_relay_destroy(abcdk_rtsp_relay_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return;
}

abcdk_rtsp_relay_t *abcdk_rtsp_relay_create(abcdk_rtsp_server_t *server_ctx, const char *server_media, const char *src_url, const char *src_fmt, float src_xspeed, int src_timeout, int src_retry)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return NULL;
}

#endif // #if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)