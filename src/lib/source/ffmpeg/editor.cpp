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
    abcdk_ffmpeg_editor_param_t param;

    AVDictionary *option_ctx;
    AVFormatContext *media_ctx;
    AVIOContext *vio_ctx;

    /*最新活动时间(微秒).*/
    uint64_t latest_time;

    /**读, 流过滤器. */
    std::map<int, abcdk_ffmpeg_bsf_t *> read_bsf_list;
    int read_bsf_list_pos;

    /**读, 第一帧DTS出现时间(微秒).*/
    std::map<int, uint64_t> read_dts_first_time;

    /**读, 第一帧DTS.*/
    std::map<int, int64_t> read_dts_first;

    /**读, 最近的DTS.*/
    std::map<int, int64_t> read_dts_latest;

    /**读, 是否已经到末尾. */
    int read_packet_eof;
    int read_frame_eof;

    /**写, 最近的DTS.*/
    std::map<int, int64_t> write_dts_latest;

    /**写, 源时间基值.*/
    std::map<int, AVRational> write_src_time_base;

    /**写, 头部写入是否成功. */
    int write_header_ok;

    /**写, 尾部写入是否成功. */
    int write_trailer_ok;

}; // abcdk_ffmpeg_editor_t;

void abcdk_ffmpeg_editor_free(abcdk_ffmpeg_editor_t **ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return ;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_editor_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    // 作者, 写入末尾.
    if (ctx_p->writer)
        abcdk_ffmpeg_editor_write_trailer(ctx_p);

    abcdk_heap_freep((void **)&ctx_p->param.fmt);
    abcdk_heap_freep((void **)&ctx_p->param.url);

    av_dict_free(&ctx_p->option_ctx);
    abcdk_ffmpeg_media_free(&ctx_p->media_ctx);
    abcdk_ffmpeg_io_free(&ctx_p->vio_ctx);

    for (auto &one : ctx_p->read_bsf_list)
        abcdk_ffmpeg_bsf_free(&one.second);

    ctx_p->read_bsf_list.clear();

    delete ctx_p;
#endif //#ifndef HAVE_FFMPEG
}

abcdk_ffmpeg_editor_t *abcdk_ffmpeg_editor_alloc(int writer)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return NULL;
#else //#ifndef HAVE_FFMPEG
    abcdk_ffmpeg_editor_t *ctx;

    ctx = new abcdk_ffmpeg_editor_t;
    if (!ctx)
        return NULL;

    ctx->writer = writer;
    ctx->option_ctx = NULL;
    ctx->media_ctx = NULL;
    ctx->vio_ctx = NULL;
    ctx->latest_time = 0;
    ctx->read_bsf_list_pos = 0;
    ctx->read_packet_eof = 0;
    ctx->read_frame_eof = 0;
    ctx->write_header_ok = 0;
    ctx->write_trailer_ok = 0;

    return ctx;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_editor_stream_nb(abcdk_ffmpeg_editor_t *ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    double scale;

    assert(ctx != NULL);
    assert(ctx->media_ctx != NULL);

    return ctx->media_ctx->nb_streams;
#endif // #ifndef HAVE_FFMPEG
}

AVStream *abcdk_ffmpeg_editor_stream_ctx(abcdk_ffmpeg_editor_t *ctx, int stream)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return NULL;
#else //#ifndef HAVE_FFMPEG
    assert(ctx != NULL && stream >= 0);
    assert(ctx->media_ctx != NULL);
    assert(ctx->media_ctx->nb_streams > stream);

    return ctx->media_ctx->streams[stream];
#endif // #ifndef HAVE_FFMPEG
}


double abcdk_ffmpeg_editor_stream_timebase_q2d(abcdk_ffmpeg_editor_t *ctx, int stream)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return 0.0;
#else //#ifndef HAVE_FFMPEG
    assert(ctx != NULL && stream >= 0);
    assert(ctx->media_ctx != NULL);
    assert(ctx->media_ctx->nb_streams > stream);

    if (ctx->writer || ctx->param.read_rate_scale <= 0)
        return abcdk_ffmpeg_stream_timebase_q2d(ctx->media_ctx->streams[stream], 1.0);
    else
        return abcdk_ffmpeg_stream_timebase_q2d(ctx->media_ctx->streams[stream], (double)ctx->param.read_rate_scale / 1000.);
#endif // #ifndef HAVE_FFMPEG  
}


double abcdk_ffmpeg_editor_stream_duration(abcdk_ffmpeg_editor_t *ctx, int stream)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return 0.0;
#else //#ifndef HAVE_FFMPEG
    assert(ctx != NULL && stream >= 0);
    assert(ctx->media_ctx != NULL);
    assert(ctx->media_ctx->nb_streams > stream);

    if (ctx->writer || ctx->param.read_rate_scale <= 0)
        return abcdk_ffmpeg_stream_duration(ctx->media_ctx->streams[stream], 1.0);
    else
        return abcdk_ffmpeg_stream_duration(ctx->media_ctx->streams[stream], (double)ctx->param.read_rate_scale / 1000.);
#endif // #ifndef HAVE_FFMPEG  
}


double abcdk_ffmpeg_editor_stream_time2rate(abcdk_ffmpeg_editor_t *ctx, int stream)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return 0.000001;
#else //#ifndef HAVE_FFMPEG
    assert(ctx != NULL && stream >= 0);
    assert(ctx->media_ctx != NULL);
    assert(ctx->media_ctx->nb_streams > stream);

    if (ctx->writer || ctx->param.read_rate_scale <= 0)
        return abcdk_ffmpeg_stream_time2rate(ctx->media_ctx->streams[stream], 1.0);
    else
        return abcdk_ffmpeg_stream_time2rate(ctx->media_ctx->streams[stream], (double)ctx->param.read_rate_scale / 1000.);
#endif // #ifndef HAVE_FFMPEG  
}

double abcdk_ffmpeg_editor_stream_ts2sec(abcdk_ffmpeg_editor_t *ctx, int stream, int64_t ts)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -0.000001;
#else //#ifndef HAVE_FFMPEG
    assert(ctx != NULL && stream >= 0);
    assert(ctx->media_ctx != NULL);
    assert(ctx->media_ctx->nb_streams > stream);

    if (ctx->writer || ctx->param.read_rate_scale <= 0)
        return abcdk_ffmpeg_stream_ts2sec(ctx->media_ctx->streams[stream], ts, 1.0);
    else
        return abcdk_ffmpeg_stream_ts2sec(ctx->media_ctx->streams[stream], ts, (double)ctx->param.read_rate_scale / 1000.);
#endif // #ifndef HAVE_FFMPEG
}

int64_t abcdk_ffmpeg_editor_stream_ts2num(abcdk_ffmpeg_editor_t *ctx, int stream, int64_t ts)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    assert(ctx != NULL && stream >= 0);
    assert(ctx->media_ctx != NULL);
    assert(ctx->media_ctx->nb_streams > stream);

    if (ctx->writer || ctx->param.read_rate_scale <= 0)
        return abcdk_ffmpeg_stream_ts2num(ctx->media_ctx->streams[stream], ts, 1.0);
    else
        return abcdk_ffmpeg_stream_ts2num(ctx->media_ctx->streams[stream], ts, (double)ctx->param.read_rate_scale / 1000.);
#endif // #ifndef HAVE_FFMPEG
}

#ifdef HAVE_FFMPEG

static int _abcdk_ffmpeg_editor_interrupt_cb(void *args)
{
    abcdk_ffmpeg_editor_t *ctx = (abcdk_ffmpeg_editor_t *)args;
    uint64_t check_time = abcdk_time_systime(6);

    // 允许未启用.
    if (ctx->param.timeout <= 0)
        return 0;

    // 如果是作者并且已经连接成功, 则忽略超时检测.
    if (ctx->writer && ctx->write_header_ok)
        return 0;

    // 超时检测, 未超时返回0, 否则返回-1.
    if ((check_time - ctx->latest_time) < ctx->param.timeout * 1000000)
        return 0;

    return -1;
}

static int _abcdk_avformat_media_init(abcdk_ffmpeg_editor_t *ctx)
{
    AVInputFormat *fmt_ctx = NULL;
    AVFormatContext *media_ctx = NULL;
    int chk;

    // free old context.
    abcdk_ffmpeg_media_free(&ctx->media_ctx);

    ctx->media_ctx = avformat_alloc_context();
    if (!ctx->media_ctx)
        return AVERROR(ENOMEM);

    // 更新时间, 否则重复打开时会有超时发生.
    ctx->latest_time = abcdk_time_systime(6);

    ctx->media_ctx->interrupt_callback.callback = _abcdk_ffmpeg_editor_interrupt_cb;
    ctx->media_ctx->interrupt_callback.opaque = ctx;

    if (ctx->vio_ctx)
    {
        ctx->media_ctx->pb = ctx->vio_ctx;
        ctx->media_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
    }

    if (ctx->writer)
    {
        if (ctx->param.write_nodelay)
            ctx->media_ctx->flags |= AVFMT_FLAG_FLUSH_PACKETS;

        ctx->media_ctx->oformat = av_guess_format(ctx->param.fmt, ctx->param.url, NULL);
        if (!ctx->media_ctx->oformat)
        {
            abcdk_ffmpeg_media_free(&ctx->media_ctx);
            return AVERROR_MUXER_NOT_FOUND;
        }

#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
        ctx->media_ctx->url = av_strdup(ctx->param.url);
#if FF_API_FORMAT_FILENAME
        strncpy(ctx->media_ctx->filename, ctx->param.url, sizeof(ctx->media_ctx->filename));
#endif // #if FF_API_FORMAT_FILENAME
#else  // #if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
        strncpy(ctx->media_ctx->filename, ctx->param.url, sizeof(ctx->media_ctx->filename));
#endif // #if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
    }
    else
    {
        /*
         * 1: 如果不知道下面标志如何使用，一定不要附加这个标志。
         * 2: 如果附加此标志，会造成数据流开头的数据包丢失(N个)。
         * 3: 如果未附加此标志，网络流会产生不确定的延时(N毫秒~N秒)。
         */
        // ctx->media_ctx->flags |= AVFMT_FLAG_NOBUFFER;

        fmt_ctx = (AVInputFormat *)av_find_input_format(ctx->param.fmt);
        chk = avformat_open_input(&ctx->media_ctx, ctx->param.url, fmt_ctx, &ctx->option_ctx);
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

        // 打印媒体信息.
        abcdk_ffmpeg_media_dump(ctx->media_ctx, 0);
    }

    return 0;
}

static int _abcdk_avformat_media_init_bsf(abcdk_ffmpeg_editor_t *ctx)
{
    const char *name_p;
    AVStream *vs_ctx_p;
    int is_mp4_file = 0;

    if (ctx->writer)
    {
        ; // nothing.
    }
    else if (ctx->param.read_mp4toannexb)
    {
        is_mp4_file |= (strcmp(ctx->media_ctx->iformat->long_name, "QuickTime / MOV") == 0 ? 0x1 : 0);
        is_mp4_file |= (strcmp(ctx->media_ctx->iformat->long_name, "FLV (Flash Video)") == 0 ? 0x2 : 0);
        is_mp4_file |= (strcmp(ctx->media_ctx->iformat->long_name, "Matroska / WebM") == 0 ? 0x4 : 0);

        for (int i = 0; i < ctx->media_ctx->nb_streams; i++)
        {
            name_p = NULL;
            vs_ctx_p = ctx->media_ctx->streams[i];

            if (is_mp4_file && vs_ctx_p->codecpar->codec_id == AV_CODEC_ID_H264)
                name_p = "h264_mp4toannexb";
            else if (is_mp4_file && vs_ctx_p->codecpar->codec_id == AV_CODEC_ID_H265)
                name_p = "hevc_mp4toannexb";
            else
                name_p = "fifo";

            ctx->read_bsf_list[vs_ctx_p->index] = abcdk_ffmpeg_bsf_alloc(name_p);

            if (!ctx->read_bsf_list[vs_ctx_p->index])
                return AVERROR_FILTER_NOT_FOUND;

            if (abcdk_strcmp(name_p, "fifo", 1) != 0)
                abcdk_ffmpeg_bsf_init2(ctx->read_bsf_list[vs_ctx_p->index], vs_ctx_p->codecpar);
        }
    }

    return 0;
}

static int _abcdk_avformat_media_init_ts(abcdk_ffmpeg_editor_t *ctx)
{
    if (ctx->writer)
    {
        for (int i = 0; i < ctx->media_ctx->nb_streams; i++)
        {
            ctx->write_dts_latest[i] = AV_NOPTS_VALUE;
        }
    }
    else
    {
        for (int i = 0; i < ctx->media_ctx->nb_streams; i++)
        {
            ctx->read_dts_first[i] = AV_NOPTS_VALUE;
            ctx->read_dts_latest[i] = AV_NOPTS_VALUE;
            ctx->read_dts_first_time[i] = 0;
        }
    }

    return 0;
}

#endif //#ifdef HAVE_FFMPEG

int abcdk_ffmpeg_editor_open(abcdk_ffmpeg_editor_t *ctx, const abcdk_ffmpeg_editor_param_t *param)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL && param != NULL);

    ctx->param = *param;

    // 覆盖外部指针.
    ctx->param.fmt = abcdk_strdup_safe(param->fmt);
    ctx->param.url = abcdk_strdup_safe(param->url);

    if (ctx->param.url)
    {
        if (ctx->writer)
        {
            if (!ctx->param.fmt)
            {
                // 下面的网络协议, 如果未指定格式, 需要内部自行判定.
                if (abcdk_strncmp(ctx->param.url, "rtsp://", 7, 0) == 0)
                    ctx->param.fmt = abcdk_strdup_safe("rtsp");
                else if (abcdk_strncmp(ctx->param.url, "rtsps://", 8, 0) == 0)
                    ctx->param.fmt = abcdk_strdup_safe("rtsp");
                else if (abcdk_strncmp(ctx->param.url, "rtmp://", 7, 0) == 0)
                    ctx->param.fmt = abcdk_strdup_safe("flv");
                else if (abcdk_strncmp(ctx->param.url, "rtmps://", 8, 0) == 0)
                    ctx->param.fmt = abcdk_strdup_safe("flv");
            }

            chk = _abcdk_avformat_media_init(ctx);
            if (chk < 0)
                return chk;
        }
        else
        {
            av_dict_set_int(&ctx->option_ctx, "stimeout", ctx->param.timeout * 1000000, 0);   // rtsp
            av_dict_set_int(&ctx->option_ctx, "rw_timeout", ctx->param.timeout * 1000000, 0); // rtmp

            if (ctx->param.rtsp_transport)
            {
                if (ctx->param.rtsp_transport == 1)
                    av_dict_set(&ctx->option_ctx, "rtsp_transport", "udp", 0);
                else if (ctx->param.rtsp_transport == 2)
                    av_dict_set(&ctx->option_ctx, "rtsp_transport", "tcp", 0);
            }

            chk = _abcdk_avformat_media_init(ctx);
            if (chk < 0)
                return chk;
        }
    }
    else if (ctx->param.vio.read_cb || ctx->param.vio.write_cb)
    {
        ctx->vio_ctx = abcdk_ffmpeg_io_alloc(8, ctx->writer);
        if (!ctx->vio_ctx)
            return AVERROR(ENOMEM);

        ctx->vio_ctx->opaque = ctx->param.vio.opaque;
        ctx->vio_ctx->read_packet = ctx->param.vio.read_cb;
        ctx->vio_ctx->write_packet = ctx->param.vio.write_cb;

        chk = _abcdk_avformat_media_init(ctx);
        if (chk < 0)
            return chk;

        ctx->vio_ctx = NULL; // It has been hosted and can no longer be used.
    }
    else
    {
        return AVERROR(EINVAL);
    }

    chk = _abcdk_avformat_media_init_bsf(ctx);
    if (chk < 0)
        return chk;

    chk = _abcdk_avformat_media_init_ts(ctx);
    if (chk < 0)
        return chk;

    return 0;
#endif // #ifndef HAVE_FFMPEG
}

#ifdef HAVE_FFMPEG

/**
 * 延时检查。
 *
 * @return 0 未满足，!0 已满足。
 */
static int _abcdk_ffmpeg_editor_read_delay_check(abcdk_ffmpeg_editor_t *ctx, int stream)
{
    double a1, a2, a, b;

    if (ctx->param.read_nodelay)
        return 1;

    /*如果是无效的DTS则直接返回已满足(!0).*/
    if (ctx->read_dts_latest[stream] == (int64_t)AV_NOPTS_VALUE)
        return 1;

    a1 = abcdk_ffmpeg_editor_stream_ts2sec(ctx, stream, ctx->read_dts_first[stream]);
    a2 = abcdk_ffmpeg_editor_stream_ts2sec(ctx, stream, ctx->read_dts_latest[stream]);

    /*
     * 1：计算当前帧与第一帧的时间差.
     * 2：因为流的起始值可能不为零(或为负, 或为正), 所以时间轴调整为从零开始以便于计算延时.
     */

    a = (a2 - a1) - (a1 - a1);
    b = (double)(abcdk_time_systime(6) - ctx->read_dts_first_time[stream]) / 1000000.;

    return (a >= b ? 0 : 1);
}

static void _abcdk_ffmpeg_editor_read_delay(abcdk_ffmpeg_editor_t *ctx)
{
    AVStream *vs_ctx_p = NULL;
    int blocks = 0;

next_delay:

    /*如果已经超时则直接返回. */
    if (_abcdk_ffmpeg_editor_interrupt_cb(ctx) != 0)
        return;

    blocks = 0;
    for (int i = 0; i < ctx->media_ctx->nb_streams; i++)
    {
        vs_ctx_p = ctx->media_ctx->streams[i];

        /*检测延时是否已经满足.*/
        if (!_abcdk_ffmpeg_editor_read_delay_check(ctx, vs_ctx_p->index))
            blocks += 1;
    }

    /*以最慢流的为基准.*/
    if (blocks > 0)
    {
        usleep(1000); // 1000fps
        goto next_delay;
    }

    return;
}

static int _abcdk_ffmpeg_editor_bsf_send(abcdk_ffmpeg_editor_t *ctx, AVPacket *src)
{
    return abcdk_ffmpeg_bsf_send(ctx->read_bsf_list[src->stream_index], src);
}

static int _abcdk_ffmpeg_editor_bsf_recv(abcdk_ffmpeg_editor_t *ctx, AVPacket *dst)
{
    std::map<int, abcdk_ffmpeg_bsf_t *>::iterator bsf_it;
    int chk;

    // 更新最新的活动时间, 不然会超时.
    ctx->latest_time = abcdk_time_systime(6);

    for (int i = 0; i < ctx->read_bsf_list.size(); i++)
    {
        chk = abcdk_ffmpeg_bsf_recv(ctx->read_bsf_list[ctx->read_bsf_list_pos], dst);

        // 循环滚动游标.
        ctx->read_bsf_list_pos += 1;
        ctx->read_bsf_list_pos = ctx->read_bsf_list_pos % ctx->read_bsf_list.size();

        if (chk == 0)
            continue;
        else if (chk > 0)
            return 0;
        else
            return AVERROR(EPIPE);
    }

    if (ctx->read_packet_eof)
    {
        ctx->read_packet_eof = 2;
        return AVERROR_EOF;
    }
    else
    {
        return AVERROR(EAGAIN);
    }
}

static int _abcdk_ffmpeg_editor_packet_recv(abcdk_ffmpeg_editor_t *ctx, AVPacket *dst)
{
    AVStream *vs_ctx_p = NULL;
    int chk;

    // 返回重试, 如果已经到末尾并且已启用流过滤.
    if (ctx->read_packet_eof && ctx->param.read_mp4toannexb)
        return AVERROR(EAGAIN);

    // 返回结束, 如果已经到末尾并且未启动流过滤.
    if (ctx->read_packet_eof && !ctx->param.read_mp4toannexb)
        return AVERROR_EOF;

    // 更新最新的活动时间, 不然会超时.
    ctx->latest_time = abcdk_time_systime(6);

    chk = av_read_frame(ctx->media_ctx, dst);
    if (chk == AVERROR_EOF)
    {
        ctx->read_packet_eof = 1;
        return (ctx->param.read_mp4toannexb ? AVERROR(EAGAIN) : AVERROR_EOF);
    }
    else if (chk < 0)
    {
        return chk;
    }

    vs_ctx_p = ctx->media_ctx->streams[dst->stream_index];

    /*记录有效的DTS, 并且记录开始读取时间(用于记算拉流延时).*/
    if (dst->dts != (int64_t)AV_NOPTS_VALUE)
    {
        /*记录当前的DTS。*/
        ctx->read_dts_latest[dst->stream_index] = dst->dts;

        /*
         * 满足以下两个条件时, 需要更新时间轴开始时间.
         * 1：开始时间无效时.
         * 2：时间轴重置.
         */
        if (ctx->read_dts_first[dst->stream_index] == (int64_t)AV_NOPTS_VALUE ||
            ctx->read_dts_first[dst->stream_index] > ctx->read_dts_latest[dst->stream_index])
        {
            ctx->read_dts_first[dst->stream_index] = ctx->read_dts_latest[dst->stream_index];
            ctx->read_dts_first_time[dst->stream_index] = abcdk_time_systime(6);
        }
    }

    if (ctx->param.read_ignore_video && vs_ctx_p->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        av_packet_unref(dst);
        return AVERROR(EAGAIN);
    }
    else if (ctx->param.read_ignore_audio && vs_ctx_p->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
    {
        av_packet_unref(dst);
        return AVERROR(EAGAIN);
    }
    else if (ctx->param.read_ignore_subtitle && vs_ctx_p->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)
    {
        av_packet_unref(dst);
        return AVERROR(EAGAIN);
    }
    else if (vs_ctx_p->codecpar->codec_type != AVMEDIA_TYPE_VIDEO &&
             vs_ctx_p->codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
             vs_ctx_p->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE)
    {
        av_packet_unref(dst);
        return AVERROR(EAGAIN);
    }

    return 0;
}

#endif //#ifdef HAVE_FFMPEG

int abcdk_ffmpeg_editor_read_packet(abcdk_ffmpeg_editor_t *ctx, AVPacket *dst)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int need_delay = 1;
    int chk;

    assert(ctx != NULL && dst != NULL);
    assert(!ctx->writer);
    assert(ctx->media_ctx);

    // Clean up data that is no longer used.
    av_packet_unref(dst);

next_packet:

    if (need_delay)
    {
        // 等待下一帧时间到达.
        _abcdk_ffmpeg_editor_read_delay(ctx);
        need_delay = 0;
    }

    if (ctx->param.read_mp4toannexb)
    {
        chk = _abcdk_ffmpeg_editor_bsf_recv(ctx, dst);
        if (chk == 0)
            return 0;

        if (chk == AVERROR(EAGAIN))
            chk = _abcdk_ffmpeg_editor_packet_recv(ctx, dst);
    }
    else
    {
        chk = _abcdk_ffmpeg_editor_packet_recv(ctx, dst);
    }

    if (chk == AVERROR(EAGAIN))
        goto next_packet;
    else if (chk < 0)
        return chk;

    if (ctx->param.read_mp4toannexb)
    {
        _abcdk_ffmpeg_editor_bsf_send(ctx, dst); // auto unref.
        goto next_packet;
    }

    return 0;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_editor_add_stream(abcdk_ffmpeg_editor_t *ctx, const AVCodecContext *opt)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    AVCodec *codec_ctx_p = NULL;
    AVStream *vs_ctx = NULL;
    int chk;

    assert(ctx != NULL && opt != NULL);
    assert(ctx->writer);
    assert(ctx->media_ctx);

    codec_ctx_p = avcodec_find_encoder(opt->codec_id);
    if (!codec_ctx_p)
        return AVERROR_ENCODER_NOT_FOUND;

    vs_ctx = avformat_new_stream(ctx->media_ctx, codec_ctx_p);
    if (!vs_ctx)
        return AVERROR(ENOMEM);

    chk = avcodec_parameters_from_context(vs_ctx->codecpar, opt);
    if (chk < 0)
        return chk;

    vs_ctx->codecpar->codec_tag = 0; // 不能复用.

    vs_ctx->time_base = opt->time_base;
    vs_ctx->avg_frame_rate = vs_ctx->r_frame_rate = opt->framerate;

    /*下面的也要复制, 因为一些定制的ffmpeg未完成启用新的参数. */
#if FF_API_LAVF_AVCTX
    chk = avcodec_parameters_to_context(vs_ctx->codec, vs_ctx->codecpar);
    if (chk < 0)
        return chk;

    vs_ctx->codec->codec_tag = 0; // 不能复用.

    vs_ctx->codec->time_base = opt->time_base;
    vs_ctx->codec->framerate = opt->framerate;

    /*如果流需要设置全局头部, 则编码器需要知道这个请求. */
    if (ctx->media_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        vs_ctx->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
#endif // FF_API_LAVF_AVCTX

    abcdk_ffmpeg_stream_fix_bitrate(vs_ctx);

    //记录源时间基值.
    ctx->write_src_time_base[vs_ctx->index] = opt->time_base;

    return vs_ctx->index;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_editor_add_stream2(abcdk_ffmpeg_editor_t *ctx, const AVCodecParameters *opt, const AVRational *time_base, 
                                    const AVRational *avg_frame_rate,const AVRational *r_frame_rate)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    AVCodec *codec_ctx_p = NULL;
    AVStream *vs_ctx = NULL;
    int chk;

    assert(ctx != NULL && opt != NULL && time_base != NULL);
    assert(ctx->writer);
    assert(ctx->media_ctx);

    codec_ctx_p = avcodec_find_encoder(opt->codec_id);
    if (!codec_ctx_p)
        return AVERROR_ENCODER_NOT_FOUND;

    vs_ctx = avformat_new_stream(ctx->media_ctx, codec_ctx_p);
    if (!vs_ctx)
        return AVERROR(ENOMEM);

    chk = avcodec_parameters_copy(vs_ctx->codecpar, opt);
    if (chk < 0)
        return chk;

    vs_ctx->codecpar->codec_tag = 0; // 不能复用.
    vs_ctx->time_base = *time_base;
    vs_ctx->avg_frame_rate = (avg_frame_rate ? *avg_frame_rate : (r_frame_rate ? *r_frame_rate : av_make_q(time_base->den, time_base->num)));
    vs_ctx->r_frame_rate = (r_frame_rate ? *r_frame_rate : (avg_frame_rate ? *avg_frame_rate : av_make_q(time_base->den, time_base->num)));

    /*下面的也要复制, 因为一些定制的ffmpeg未完成启用新的参数. */
#if FF_API_LAVF_AVCTX
    chk = avcodec_parameters_to_context(vs_ctx->codec, opt);
    if (chk < 0)
        return chk;

    vs_ctx->codec->codec_tag = 0; // 不能复用.
    vs_ctx->codec->time_base = *time_base;
    vs_ctx->codec->framerate = (avg_frame_rate ? *avg_frame_rate : (r_frame_rate ? *r_frame_rate : av_make_q(time_base->den, time_base->num)));

    /*如果流需要设置全局头部, 则编码器需要知道这个请求. */
    if (ctx->media_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        vs_ctx->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
#endif // FF_API_LAVF_AVCTX

    abcdk_ffmpeg_stream_fix_bitrate(vs_ctx);

    //记录源时间基值.
    ctx->write_src_time_base[vs_ctx->index] = *time_base;

    return vs_ctx->index;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_editor_write_packet(abcdk_ffmpeg_editor_t *ctx, AVPacket *src)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    AVRational tb_src, tb_dst;
    AVStream *vs_ctx_p = NULL;
    int chk;

    assert(ctx != NULL && src != NULL);
    assert(ctx->writer);
    assert(ctx->media_ctx);
    assert(ctx->media_ctx->oformat);

    assert(src->stream_index >= 0 && src->stream_index < ctx->media_ctx->nb_streams);

    /*The body can only be written after the header is written.*/
    if (!ctx->write_header_ok)
    {
        chk = abcdk_ffmpeg_editor_write_header(ctx);
        if (chk < 0)
            return chk;
    }

    vs_ctx_p = ctx->media_ctx->streams[src->stream_index];

    /*
     * 如果没有源时间基, 则使用内部时间基值.
     * 注: 如果时间基值错误, 编码数据在解码后无法正常播放, 常见的现像是DTS,PTS,DURATION异常.
     */
    tb_dst = vs_ctx_p->time_base;
    tb_src = ctx->write_src_time_base[src->stream_index];

    //设置为未知或无效.
    src->pos = -1;

    //确保DTS单调递增(简单修复).
    if (src->dts != AV_NOPTS_VALUE && src->dts <= ctx->write_dts_latest[src->stream_index])
        src->dts = ctx->write_dts_latest[src->stream_index] + 1;

    //记录最新的DTS.
    ctx->write_dts_latest[src->stream_index] = src->dts;

    //修复错误的时长(简单修复).
    if (src->duration == 0)
        src->duration = 1;

    //重置DTS,PTS,DURATION.
    av_packet_rescale_ts(src, tb_src, tb_dst);

    //确保DTS不大于PTS.
    if (src->dts > src->pts)
        src->dts = src->pts;

    //当存在多个流时需要交替写入, 否则可能无法正常播放.
    chk = (ctx->media_ctx->nb_streams > 1 ? av_interleaved_write_frame(ctx->media_ctx, src) : av_write_frame(ctx->media_ctx, src));
    if (chk != 0)
        return chk;

    if (ctx->param.write_nodelay && ctx->media_ctx->pb)
        avio_flush(ctx->media_ctx->pb);

    return 0;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_editor_write_header(abcdk_ffmpeg_editor_t *ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    char *url_p;
    int chk;

    assert(ctx != NULL);
    assert(ctx->writer);
    assert(ctx->media_ctx);
    assert(ctx->media_ctx->oformat);

    /*Just once.*/
    if (ctx->write_header_ok)
        return 0;

    if ((ctx->media_ctx->oformat->flags & AVFMT_NOFILE) || ctx->media_ctx->pb)
    {
        /*网络流或自定议IO不需要任何额外操作.*/
    }
    else
    {
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
        url_p = ctx->media_ctx->url;
#else  // #if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
        url_p = ctx->media_ctx->filename;
#endif // #if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)

        chk = avio_open(&ctx->media_ctx->pb, url_p, AVIO_FLAG_WRITE);
        if (chk < 0)
            return chk;

        if (ctx->param.rtsp_transport)
        {
            if (ctx->param.rtsp_transport == 1)
                av_dict_set(&ctx->option_ctx, "rtsp_transport", "udp", 0);
            else if (ctx->param.rtsp_transport == 2)
                av_dict_set(&ctx->option_ctx, "rtsp_transport", "tcp", 0);
        }

        if (ctx->param.write_fmp4)
            av_dict_set(&ctx->option_ctx, "movflags", "frag_keyframe+empty_moov+default_base_moof+faststart", 0);

        // 控制输出时音频和视频数据包之间的最大"交织时间差".
        av_dict_set(&ctx->option_ctx, "max_interleave_delta", "0", 0);
    }

    chk = avformat_write_header(ctx->media_ctx, &ctx->option_ctx);
    if (chk < 0)
        return chk;

    chk = _abcdk_avformat_media_init_bsf(ctx);
    if (chk < 0)
        return chk;

    chk = _abcdk_avformat_media_init_ts(ctx);
    if (chk < 0)
        return chk;

    /*it's ok.*/
    ctx->write_header_ok = 1;

    // 打印媒体信息.
    abcdk_ffmpeg_media_dump(ctx->media_ctx, 1);

    return 0;
#endif // #ifndef HAVE_FFMPEG
}

int abcdk_ffmpeg_editor_write_trailer(abcdk_ffmpeg_editor_t *ctx)
{
#ifndef HAVE_FFMPEG
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具."));
    return -1;
#else //#ifndef HAVE_FFMPEG
    int chk;

    assert(ctx != NULL);
    assert(ctx->writer);
    assert(ctx->media_ctx);
    assert(ctx->media_ctx->oformat);

    /*The end can only be written after the header is written.*/
    if (!ctx->write_header_ok)
        return AVERROR(EPERM);

    /*Just once.*/
    if (ctx->write_trailer_ok)
        return 0;

    chk = av_write_trailer(ctx->media_ctx);
    if (chk < 0)
        return chk;

    /*it's ok.*/
    ctx->write_trailer_ok = 1;

    return 0;
#endif // #ifndef HAVE_FFMPEG
}