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

    /**读, 第一帧DTS出现时间(微秒).*/
    std::map<int, uint64_t> read_dts_first_time;

    /**读, 第一帧DTS.*/
    std::map<int, int64_t> read_dts_first;

    /**读, 最近的DTS.*/
    std::map<int, int64_t> read_dts_latest;

    /**读, 是否已经到末尾. */
    int read_packet_eof;
    int read_frame_eof;

    /**写, 头部写入是否成功. */
    int write_header_ok;

    std::map<int, abcdk_ffmpeg_bsf_t *> bsf_list;
    int bsf_list_read_pos;
    std::map<int, abcdk_ffmpeg_decoder_t *> decoder_list;
    std::map<int, abcdk_ffmpeg_encoder_t *> encoder_list;

}; // abcdk_ffmpeg_editor_t;

void abcdk_ffmpeg_editor_free(abcdk_ffmpeg_editor_t **ctx)
{
    abcdk_ffmpeg_editor_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_heap_freep((void **)&ctx_p->param.fmt);
    abcdk_heap_freep((void **)&ctx_p->param.url);

    av_dict_free(&ctx_p->option_ctx);
    abcdk_ffmpeg_media_free(&ctx_p->media_ctx);
    abcdk_ffmpeg_io_free(&ctx_p->vio_ctx);

    for (auto &one : ctx_p->bsf_list)
        abcdk_ffmpeg_bsf_free(&one.second);
    for (auto &one : ctx_p->decoder_list)
        abcdk_ffmpeg_decoder_free(&one.second);
    for (auto &one : ctx_p->encoder_list)
        abcdk_ffmpeg_encoder_free(&one.second);

    ctx_p->bsf_list.clear();
    ctx_p->decoder_list.clear();
    ctx_p->encoder_list.clear();

    delete ctx_p;
}

abcdk_ffmpeg_editor_t *abcdk_ffmpeg_editor_alloc(int writer)
{
    abcdk_ffmpeg_editor_t *ctx;

    ctx = new abcdk_ffmpeg_editor_t;
    if (!ctx)
        return NULL;

    ctx->writer = writer;
    ctx->option_ctx = NULL;
    ctx->media_ctx = NULL;
    ctx->vio_ctx = NULL;
    ctx->latest_time = 0;
    ctx->read_packet_eof = 0;
    ctx->read_frame_eof = 0;
    ctx->write_header_ok = 0;
    ctx->bsf_list_read_pos = 0;

    return ctx;
}

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
        av_dict_set(&ctx->media_ctx->metadata, "service", "ABCDK", 0);
        av_dict_set(&ctx->media_ctx->metadata, "service_name", "ABCDK", 0);
        av_dict_set(&ctx->media_ctx->metadata, "service_provider", "ABCDK", 0);
        av_dict_set(&ctx->media_ctx->metadata, "artist", "ABCDK", 0);

        if (ctx->param.write_nodelay)
            ctx->media_ctx->flags |= AVFMT_FLAG_FLUSH_PACKETS;

        ctx->media_ctx->oformat = av_guess_format(ctx->param.fmt, ctx->param.url, NULL);
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
            const char *name_p = NULL;
            AVStream *stream_p = ctx->media_ctx->streams[i];

            if (is_mp4_file && stream_p->codecpar->codec_id == AV_CODEC_ID_H264)
                name_p = "h264_mp4toannexb";
            else if (is_mp4_file && stream_p->codecpar->codec_id == AV_CODEC_ID_H265)
                name_p = "hevc_mp4toannexb";
            else
                name_p = "fifo";

            ctx->bsf_list[stream_p->index] = abcdk_ffmpeg_bsf_alloc(name_p);

            if (!ctx->bsf_list[stream_p->index])
                return AVERROR_FILTER_NOT_FOUND;

            if (abcdk_strncmp(name_p, "fifo", 4, 1) != 0)
                abcdk_ffmpeg_bsf_init2(ctx->bsf_list[stream_p->index], stream_p->codecpar);
        }
    }

    return 0;
}

static int _abcdk_avformat_media_init_ts(abcdk_ffmpeg_editor_t *ctx)
{
    for (int i = 0; i < ctx->media_ctx->nb_streams; i++)
    {
        ctx->read_dts_first[i] = (int64_t)AV_NOPTS_VALUE;
        ctx->read_dts_latest[i] = (int64_t)AV_NOPTS_VALUE;
        ctx->read_dts_first_time[i] = 0;
    }

    return 0;
}

int abcdk_ffmpeg_editor_open(abcdk_ffmpeg_editor_t *ctx, abcdk_ffmpeg_editor_param_t *param)
{
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

            /* RTSP默认走TCP，可以减少丢包。*/
            if (strncmp(ctx->param.url, "rtsp://", 7) == 0 ||
                strncmp(ctx->param.url, "rtsps://", 8) == 0)
            {
                av_dict_set(&ctx->option_ctx, "rtsp_transport", "tcp", 0);
                chk = _abcdk_avformat_media_init(ctx);
                if (chk < 0)
                {
                    av_dict_set(&ctx->option_ctx, "rtsp_transport", "udp", 0);
                    chk = _abcdk_avformat_media_init(ctx);
                    if (chk < 0)
                        return chk;
                }
            }
            else
            {
                chk = _abcdk_avformat_media_init(ctx);
                if (chk < 0)
                    return chk;
            }
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
}

double abcdk_ffmpeg_editor_ts2sec(abcdk_ffmpeg_editor_t *ctx, int stream, int64_t ts)
{
    double scale;

    assert(ctx != NULL && stream >= 0);
    assert(ctx->media_ctx != NULL);
    assert(ctx->media_ctx->nb_streams > stream);

    if(ctx->media_ctx->iformat)
        scale = (ctx->param.read_speed_scale > 0 ? (double)ctx->param.read_speed_scale / 1000. : 1.0);
    else 
        scale = 1.0;

    return abcdk_ffmpeg_stream_ts2sec(ctx->media_ctx->streams[stream], ts, scale);
}

/**
 * 延时检查。
 *
 * @return 0 未满足，!0 已满足。
 */
static int _abcdk_ffmpeg_editor_read_delay_check(abcdk_ffmpeg_editor_t *ctx, int stream)
{
    double a1, a2, a, b;

    if (ctx->param.read_speed_scale <= 0)
        return 0;

    /*如果是无效的DTS则直接返回已满足(!0).*/
    if (ctx->read_dts_latest[stream] == (int64_t)AV_NOPTS_VALUE)
        return 1;

    a1 = abcdk_ffmpeg_editor_ts2sec(ctx, stream, ctx->read_dts_first[stream]);
    a2 = abcdk_ffmpeg_editor_ts2sec(ctx, stream, ctx->read_dts_latest[stream]);

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

static int _abcdk_ffmpeg_editor_bsf_send(abcdk_ffmpeg_editor_t *ctx, AVPacket *packet)
{
    return abcdk_ffmpeg_bsf_send(ctx->bsf_list[packet->stream_index], packet);
}

static int _abcdk_ffmpeg_editor_bsf_recv(abcdk_ffmpeg_editor_t *ctx, AVPacket *packet)
{
    std::map<int, abcdk_ffmpeg_bsf_t *>::iterator bsf_it;
    int chk;

    //更新最新的活动时间, 不然会超时.
    ctx->latest_time = abcdk_time_systime(6);

    for (int i = 0; i < ctx->bsf_list.size(); i++)
    {
        chk = abcdk_ffmpeg_bsf_recv(ctx->bsf_list[ctx->bsf_list_read_pos], packet);

        //循环滚动游标.
        ctx->bsf_list_read_pos += 1;
        ctx->bsf_list_read_pos = ctx->bsf_list_read_pos % ctx->bsf_list.size();

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

static int _abcdk_ffmpeg_editor_packet_recv(abcdk_ffmpeg_editor_t *ctx, AVPacket *packet)
{
    AVStream *vs_ctx_p = NULL;
    int chk;

    // 返回重试, 如果已经到末尾并且已启用流过滤.
    if (ctx->read_packet_eof && ctx->param.read_mp4toannexb)
        return AVERROR(EAGAIN);

    // 返回结束, 如果已经到末尾并且未启动流过滤.
    if (ctx->read_packet_eof && !ctx->param.read_mp4toannexb)
        return AVERROR_EOF;
    
    //更新最新的活动时间, 不然会超时.
    ctx->latest_time = abcdk_time_systime(6);

    chk = av_read_frame(ctx->media_ctx, packet);
    if (chk == AVERROR_EOF)
    {
        ctx->read_packet_eof = 1;
        return (ctx->param.read_mp4toannexb ? AVERROR(EAGAIN) : AVERROR_EOF);
    }
    else if( chk < 0)
    {
        return chk;
    }

    vs_ctx_p = ctx->media_ctx->streams[packet->stream_index];

    /*记录有效的DTS, 并且记录开始读取时间(用于记算拉流延时).*/
    if (packet->dts != (int64_t)AV_NOPTS_VALUE)
    {
        /*记录当前的DTS。*/
        ctx->read_dts_latest[packet->stream_index] = packet->dts;

        /*
         * 满足以下两个条件时, 需要更新时间轴开始时间.
         * 1：开始时间无效时.
         * 2：时间轴重置.
         */
        if (ctx->read_dts_first[packet->stream_index] == (int64_t)AV_NOPTS_VALUE ||
            ctx->read_dts_first[packet->stream_index] > ctx->read_dts_latest[packet->stream_index])
        {
            ctx->read_dts_first[packet->stream_index] = ctx->read_dts_latest[packet->stream_index];
            ctx->read_dts_first_time[packet->stream_index] = abcdk_time_systime(6);
        }
    }

    if (ctx->param.read_ignore_video && vs_ctx_p->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        av_packet_unref(packet);
        return AVERROR(EAGAIN);
    }
    else if (ctx->param.read_ignore_audio && vs_ctx_p->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
    {
        av_packet_unref(packet);
        return AVERROR(EAGAIN);
    }
    else if (ctx->param.read_ignore_subtitle && vs_ctx_p->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)
    {
        av_packet_unref(packet);
        return AVERROR(EAGAIN);
    }
    else if (vs_ctx_p->codecpar->codec_type != AVMEDIA_TYPE_VIDEO &&
             vs_ctx_p->codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
             vs_ctx_p->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE)
    {
        av_packet_unref(packet);
        return AVERROR(EAGAIN);
    }

    return 0;
}

int abcdk_ffmpeg_editor_read_packet(abcdk_ffmpeg_editor_t *ctx, AVPacket *packet)
{
    int need_delay = 1;
    int chk;

next_packet:
    
    if(need_delay)
    {
        //等待下一帧时间到达.
        _abcdk_ffmpeg_editor_read_delay(ctx);
        need_delay = 0;
    }

    if(ctx->param.read_mp4toannexb)
    {
        chk = _abcdk_ffmpeg_editor_bsf_recv(ctx, packet);
        if (chk == 0)
            return 0;

        if(chk == AVERROR(EAGAIN))
            chk = _abcdk_ffmpeg_editor_packet_recv(ctx,packet);
    }
    else
    {
        chk = _abcdk_ffmpeg_editor_packet_recv(ctx,packet);
    }

    if (chk == AVERROR(EAGAIN))
        goto next_packet;
    else if (chk < 0)
        return chk;

    if (ctx->param.read_mp4toannexb)
    {
        _abcdk_ffmpeg_editor_bsf_send(ctx, packet);//auto unref.
        goto next_packet;
    }

    return 0;
}

static int _abcdk_ffmpeg_editor_read_decode(abcdk_ffmpeg_editor_t *ctx, AVPacket *packet, AVFrame *frame)
{
    abcdk_ffmpeg_decoder_t *decoder_ctx_p = NULL;
    AVStream *vs_ctx_p = NULL;
    int chk;

    vs_ctx_p = ctx->media_ctx->streams[packet->stream_index];

    if (!ctx->decoder_list[packet->stream_index])
    {
        ctx->decoder_list[packet->stream_index] = abcdk_ffmpeg_decoder_alloc();
        if (!ctx->decoder_list[packet->stream_index])
            return AVERROR(ENOMEM);

        chk = abcdk_ffmpeg_decoder_init3(ctx->decoder_list[packet->stream_index], vs_ctx_p->codecpar->codec_id, vs_ctx_p->codecpar, 0);
        if (chk != 0)
            return chk;
    }

    decoder_ctx_p = ctx->decoder_list[packet->stream_index];
    if (!decoder_ctx_p)
        return AVERROR_DECODER_NOT_FOUND;

    abcdk_ffmpeg_decoder_send(decoder_ctx_p, packet);

    chk = abcdk_ffmpeg_decoder_recv(decoder_ctx_p, frame);
    if (chk == 0)
        return AVERROR(EAGAIN);
    else if (chk > 0)
        return 0;
    else
        return AVERROR_EOF;
}

int abcdk_ffmpeg_editor_read_frame(abcdk_ffmpeg_editor_t *ctx, AVFrame *frame, int *stream)
{
    AVPacket *packet = NULL;
    int chk;

next_packet:

    packet = av_packet_alloc();
    if (!packet)
        return AVERROR(ENOMEM);

    chk = abcdk_ffmpeg_editor_read_packet(ctx, packet);
    if (chk != 0)
        return chk;

    if(stream)
        *stream = packet->stream_index;

    chk = _abcdk_ffmpeg_editor_read_decode(ctx, packet, frame);
    av_packet_free(&packet);

    if(chk == AVERROR(EAGAIN))
        goto next_packet;

    return chk;
}