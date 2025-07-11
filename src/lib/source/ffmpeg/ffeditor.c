/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/ffeditor.h"

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"


/**简的视音编辑。*/
struct _abcdk_ffeditor
{
    /** 配置。*/
    abcdk_ffeditor_config_t cfg;

    /** 中断。*/
    AVIOInterruptCB io_itcb;

    /** 自定义IO。*/
    AVIOContext *io_custom;

    /** 编/解码器。*/
    AVCodecContext *codec_ctx[ABCDK_FFMPEG_MAX_STREAMS];

    /** 编/解码器字典。*/
    AVDictionary *codec_dict[ABCDK_FFMPEG_MAX_STREAMS];

    /** 数据包过滤器。*/
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
    AVBSFContext *vs_filter[ABCDK_FFMPEG_MAX_STREAMS];
#else
    AVBitStreamFilterContext *vs_filter[ABCDK_FFMPEG_MAX_STREAMS];
#endif

    /** 输入是否为mp4(h264)。*/
    int input_mp4_h264[ABCDK_FFMPEG_MAX_STREAMS];
    /** 输入是否为mp4(h265)。*/
    int input_mp4_h265[ABCDK_FFMPEG_MAX_STREAMS];
    /** 输入是否为mp4(mpeg4)。*/
    int input_mp4_mpeg4[ABCDK_FFMPEG_MAX_STREAMS];

    /** 读开始时间(系统时间，微秒)*/
    uint64_t read_start[ABCDK_FFMPEG_MAX_STREAMS];

    /** 读第一帧DTS。*/
    int64_t read_dts_first[ABCDK_FFMPEG_MAX_STREAMS];

    /** 读当前DTS。*/
    int64_t read_dts[ABCDK_FFMPEG_MAX_STREAMS];

    /** 读最近KEY帧的时间(系统时间，微秒)。 */
    uint64_t read_key_ns[ABCDK_FFMPEG_MAX_STREAMS];

    /** 读最近帧分组的时间(系统时间，微秒)。 */
    uint64_t read_gop_ns[ABCDK_FFMPEG_MAX_STREAMS];

    /** 写最后DTS。*/
    int64_t write_last_dts[ABCDK_FFMPEG_MAX_STREAMS];

    /** 流容器。*/
    AVFormatContext *avctx;

    /** 流字典。*/
    AVDictionary *dict;

    /** 最近活动包时间(秒)。*/
    int64_t last_packet_time;

    /** 读缓存包.*/
    AVPacket read_pkt;

    /** 是否已经结束。*/
    int read_eof;

    /** 读流索引。 */
    int read_idx;

    /**
     * TS编号。
     *
     * 0: PTS
     * 1: DTS
     */
    int64_t ts_nums[ABCDK_FFMPEG_MAX_STREAMS][2];

    /** 是否已经写入文件头部。*/
    int write_header_ok;

    /** 是否已经写入文件尾部。*/
    int write_trailer_ok;


}; // abcdk_ffeditor_t;

static int64_t _abcdk_ffeditor_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6);
}

void abcdk_ffeditor_destroy(abcdk_ffeditor_t **ctx)
{
    abcdk_ffeditor_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*最后保证写入结尾信息。*/
    if (ctx_p->cfg.writer)
        abcdk_ffeditor_write_trailer(ctx_p);

    abcdk_avio_free(&ctx_p->io_custom);

    for (int i = 0; i < ABCDK_FFMPEG_MAX_STREAMS; i++)
    {
        if (ctx_p->codec_ctx[i])
            abcdk_avcodec_free(&ctx_p->codec_ctx[i]);

        if (ctx_p->codec_dict[i])
            av_dict_free(&ctx_p->codec_dict[i]);

        if (ctx_p->vs_filter[i])
        {
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
            av_bsf_free(&ctx_p->vs_filter[i]);
#else
            av_bitstream_filter_close(ctx_p->vs_filter[i]);
            ctx_p->vs_filter[i] = NULL;
#endif
        }
    }

    av_dict_free(&ctx_p->dict);
    abcdk_avformat_free(&ctx_p->avctx);
    av_packet_unref(&ctx_p->read_pkt);


    abcdk_heap_free(ctx_p);
}

static abcdk_ffeditor_t *_abcdk_ffeditor_alloc()
{
    abcdk_ffeditor_t *ctx = NULL;
    int chk;

    ctx = abcdk_heap_alloc(sizeof(abcdk_ffeditor_t));
    if (!ctx)
        return NULL;

    for (int i = 0; i < ABCDK_FFMPEG_MAX_STREAMS; i++)
    {
        ctx->read_dts_first[i] = (int64_t)AV_NOPTS_VALUE;
        ctx->read_dts[i] = (int64_t)AV_NOPTS_VALUE;
        ctx->read_start[i] = UINT64_MAX;
        ctx->read_key_ns[i] = UINT64_MAX;
        ctx->read_gop_ns[i] = UINT64_MAX;
        ctx->write_last_dts[i] = (int64_t)AV_NOPTS_VALUE;
    }

    av_init_packet(&ctx->read_pkt);

    return ctx;

final_error:

    abcdk_ffeditor_destroy(&ctx);

    return NULL;
}

AVFormatContext *abcdk_ffeditor_ctxptr(abcdk_ffeditor_t *ctx)
{
    assert(ctx != NULL);

    return ctx->avctx;
}

AVStream *abcdk_ffeditor_find_stream(abcdk_ffeditor_t *ctx, enum AVMediaType type)
{
    assert(ctx != NULL);

    return abcdk_avstream_find(ctx->avctx, type);
}

int abcdk_ffeditor_streams(abcdk_ffeditor_t *ctx)
{
    assert(ctx != NULL);

    return ctx->avctx->nb_streams;
}

AVStream *abcdk_ffeditor_streamptr(abcdk_ffeditor_t *ctx, int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    return ctx->avctx->streams[stream];
}

double abcdk_ffeditor_duration(abcdk_ffeditor_t *ctx, int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    if (ctx->avctx->iformat)
        return abcdk_avstream_duration(ctx->avctx, ctx->avctx->streams[stream], ctx->cfg.read_speed);

    return abcdk_avstream_duration(ctx->avctx, ctx->avctx->streams[stream], 1.0);
}

double abcdk_ffeditor_fps(abcdk_ffeditor_t *ctx, int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    if (ctx->avctx->iformat)
        return abcdk_avstream_fps(ctx->avctx, ctx->avctx->streams[stream], ctx->cfg.read_speed);

    return abcdk_avstream_fps(ctx->avctx, ctx->avctx->streams[stream], 1.0);
}

double abcdk_ffeditor_ts2sec(abcdk_ffeditor_t *ctx, int stream, int64_t ts)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    if (ctx->avctx->iformat)
        return abcdk_avstream_ts2sec(ctx->avctx, ctx->avctx->streams[stream], ts, ctx->cfg.read_speed);

    return abcdk_avstream_ts2sec(ctx->avctx, ctx->avctx->streams[stream], ts, 1.0);
}

int64_t abcdk_ffeditor_ts2num(abcdk_ffeditor_t *ctx, int stream, int64_t ts)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    if (ctx->avctx->iformat)
        return abcdk_avstream_ts2num(ctx->avctx, ctx->avctx->streams[stream], ts, ctx->cfg.read_speed);

    return abcdk_avstream_ts2num(ctx->avctx, ctx->avctx->streams[stream], ts, 1.0);
}

double abcdk_ffeditor_timebase_q2d(abcdk_ffeditor_t *ctx, int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    if (ctx->avctx->iformat)
        return abcdk_avstream_timebase_q2d(ctx->avctx, ctx->avctx->streams[stream], ctx->cfg.read_speed);

    return abcdk_avstream_timebase_q2d(ctx->avctx, ctx->avctx->streams[stream], 1.0);
}

int abcdk_ffeditor_width(abcdk_ffeditor_t *ctx, int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    return abcdk_avstream_width(ctx->avctx, ctx->avctx->streams[stream]);
}

int abcdk_ffeditor_height(abcdk_ffeditor_t *ctx, int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    return abcdk_avstream_height(ctx->avctx, ctx->avctx->streams[stream]);
}

static int _abcdk_ffeditor_interrupt_cb(void *args)
{
    abcdk_ffeditor_t *ctx = (abcdk_ffeditor_t *)args;
    uint64_t cur_time = _abcdk_ffeditor_clock();

    /*如果当前角色是作者，并且已经连接成功，超时检测忽略。*/
    if (ctx->cfg.writer && ctx->write_header_ok)
        return 0;

    if (ctx->cfg.timeout > 0)
    {
        /*超时返回-1。*/
        if ((cur_time - ctx->last_packet_time) / 1000000 >= ctx->cfg.timeout)
            return -1;
    }

    return 0;
}

static int _abcdk_ffeditor_init_capture(abcdk_ffeditor_t *ctx)
{
    int is_mp4_file = 0;
    int chk;

    if (ctx->cfg.timeout > 0)
    {
        av_dict_set_int(&ctx->dict, "stimeout", ctx->cfg.timeout * 1000000, 0);   // rtsp
        av_dict_set_int(&ctx->dict, "rw_timeout", ctx->cfg.timeout * 1000000, 0); // rtmp
    }

    ctx->avctx = abcdk_avformat_input_open(ctx->cfg.fmt, ctx->cfg.url, &ctx->io_itcb, ctx->io_custom, &ctx->dict);
    if (!ctx->avctx)
    {
        if (!ctx->cfg.url)
            return -1;

        /*如果是RTSP流，则用UDP再试一次。*/
        if (abcdk_strncmp(ctx->cfg.url, "rtsp://", 7, 0) != 0 && abcdk_strncmp(ctx->cfg.url, "rtsps://", 8, 0) != 0)
            return -1;

        av_dict_set(&ctx->dict, "rtsp_transport", "udp", 0);
        ctx->avctx = abcdk_avformat_input_open(ctx->cfg.fmt, ctx->cfg.url, &ctx->io_itcb, ctx->io_custom, &ctx->dict);
        if (!ctx->avctx)
            return -1;
    }

    /*清理托管之后的野指针。*/
    ctx->io_custom = NULL;

    chk = abcdk_avformat_input_probe(ctx->avctx, NULL);
    if (chk < 0)
        return -1;

    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "QuickTime / MOV") == 0 ? 0x1 : 0);
    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "FLV (Flash Video)") == 0 ? 0x2 : 0);
    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "Matroska / WebM") == 0 ? 0x4 : 0);

    for (int i = 0; i < ctx->avctx->nb_streams; i++)
    {
        int idx = ctx->avctx->streams[i]->index;
#if FF_API_LAVF_AVCTX
        ctx->input_mp4_h264[idx] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_H264 && is_mp4_file);
        ctx->input_mp4_h265[idx] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_HEVC && is_mp4_file);
        ctx->input_mp4_mpeg4[idx] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_MPEG4 && is_mp4_file);
#else //FF_API_LAVF_AVCTX
        ctx->input_mp4_h264[idx] = (ctx->avctx->streams[i]->codecpar->codec_id == AV_CODEC_ID_H264 && is_mp4_file);
        ctx->input_mp4_h265[idx] = (ctx->avctx->streams[i]->codecpar->codec_id == AV_CODEC_ID_HEVC && is_mp4_file);
        ctx->input_mp4_mpeg4[idx] = (ctx->avctx->streams[i]->codecpar->codec_id == AV_CODEC_ID_MPEG4 && is_mp4_file);
#endif //FF_API_LAVF_AVCTX
    }

    return 0;
}

static int _abcdk_ffeditor_init_writer(abcdk_ffeditor_t *ctx)
{
    ctx->avctx = abcdk_avformat_output_open(ctx->cfg.fmt, ctx->cfg.url, ctx->cfg.mime_type, &ctx->io_itcb, ctx->io_custom);
    if (!ctx->avctx)
        return -1;

    /*清理托管之后的野指针。*/
    ctx->io_custom = NULL;

    return 0;
}

abcdk_ffeditor_t *abcdk_ffeditor_open(abcdk_ffeditor_config_t *cfg)
{
    abcdk_ffeditor_t *ctx = NULL;
    int chk;

    assert(cfg != NULL);

    ctx = _abcdk_ffeditor_alloc();
    if (!ctx)
        return NULL;

    /*复制配置。*/
    ctx->cfg = *cfg;

    /*修复不支持的参数。*/
    ctx->cfg.io.buffer_size = ABCDK_CLAMP(ctx->cfg.io.buffer_size, (int)8, (int)1024);
    ctx->cfg.timeout = ABCDK_CLAMP(ctx->cfg.timeout, (int)-1, (int)15);
    ctx->cfg.read_speed = ABCDK_CLAMP(ctx->cfg.read_speed, (float)0.01, (float)100.0);
    ctx->cfg.read_delay_max = ABCDK_CLAMP(ctx->cfg.read_delay_max, (float)2.0, (float)86400.0);

    ctx->last_packet_time = _abcdk_ffeditor_clock();

    ctx->io_itcb.callback = _abcdk_ffeditor_interrupt_cb;
    ctx->io_itcb.opaque = ctx;

    /*按需创建自定义IO环境。*/
    if (ctx->cfg.io.read_cb || ctx->cfg.io.write_cb)
    {
        /*自定义IO必须输入格式。*/
        if(!ctx->cfg.fmt)
            goto ERR;

        /*自定义IO不支持输入URL.*/
        ctx->cfg.url = NULL;

        ctx->io_custom = abcdk_avio_alloc(ctx->cfg.io.buffer_size, ctx->cfg.writer, ctx->cfg.io.opaque);
        ctx->io_custom->read_packet = ctx->cfg.io.read_cb;
        ctx->io_custom->write_packet = ctx->cfg.io.write_cb;
    }

    if (ctx->cfg.writer)
        chk = _abcdk_ffeditor_init_writer(ctx);
    else
        chk = _abcdk_ffeditor_init_capture(ctx);

    if (chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_ffeditor_destroy(&ctx);
    return NULL;
}

int _abcdk_ffeditor_capture_open_codec(abcdk_ffeditor_t *ctx, int stream, AVCodec *codec)
{
    AVCodecContext *ctx_p = NULL;
    AVDictionary *dict_p = NULL;
    AVStream *vs_p = NULL;
    int chk;

    if (stream >= ctx->avctx->nb_streams)
        return -1;

    if (stream >= ABCDK_FFMPEG_MAX_STREAMS)
        return -1;

    /* 如果已经打开，直接返回。*/
    if (ctx->codec_ctx[stream])
        return 0;

    if (!codec)
        return -1;

    vs_p = ctx->avctx->streams[stream];

    ctx_p = abcdk_avcodec_alloc(codec);
    if (!ctx_p)
        goto final_error;

    abcdk_avstream_parameters_to_context(ctx_p, vs_p);

    ctx_p->pkt_timebase.num = vs_p->time_base.num;
    ctx_p->pkt_timebase.den = vs_p->time_base.den;

    chk = abcdk_avcodec_open(ctx_p, &dict_p);
    if (chk < 0)
        goto final_error;

    ctx->codec_ctx[stream] = ctx_p;
    ctx->codec_dict[stream] = dict_p;

    return 0;

final_error:

    abcdk_avcodec_free(&ctx_p);
    av_dict_free(&dict_p);

    return -1;
}

static int _abcdk_ffeditor_capture_codec_init(abcdk_ffeditor_t *ctx, int stream)
{
    AVStream *vs_p = NULL;
    AVCodecContext *codec_ctx_p = NULL;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    AVCodecContext *codecpar = NULL;
#else
    AVCodecParameters *codecpar = NULL;
#endif
    int chk;

    codec_ctx_p = ctx->codec_ctx[stream];
    if (codec_ctx_p)
        return 0;

    vs_p = ctx->avctx->streams[stream];
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    codecpar = vs_p->codec;
#else
    codecpar = vs_p->codecpar;
#endif

    if (ctx->cfg.try_nvcodec)
    {
        /*NVCODEC硬件解码，必须用下面的写法，因为解码器可能未安装。*/
        if (codecpar->codec_id == AV_CODEC_ID_HEVC)
            _abcdk_ffeditor_capture_open_codec(ctx, stream, abcdk_avcodec_find("hevc_cuvid", 0));
        else if (codecpar->codec_id == AV_CODEC_ID_H264)
            _abcdk_ffeditor_capture_open_codec(ctx, stream, abcdk_avcodec_find("h264_cuvid", 0));
    }

    /*如果硬件解码已经安装，到这里可能已经成功了。*/
    codec_ctx_p = ctx->codec_ctx[stream];
    if (codec_ctx_p)
        return 0;

    chk = _abcdk_ffeditor_capture_open_codec(ctx, stream, abcdk_avcodec_find2(codecpar->codec_id, 0));
    if (chk < 0)
        return -1;

    return 0;
}

/**
 * 延时检查。
 *
 * @return 0 未满足，!0 已满足。
 */
static int _abcdk_ffeditor_read_delay_check(abcdk_ffeditor_t *ctx, int stream, int flag)
{
    double a1, a2, a, b;
    int block;

    /*
     * 1：计算当前帧与第一帧的时间差。
     * 2：因为流的起始值可能不为零(或为负，或为正)，所以时间轴调整为从零开始，便于计算延时。
     */

    if (flag == 1)
    {
        if (ctx->cfg.read_flush)
            return 0;

        /*如果是无效的DTS，直接返回0。*/
        if (ctx->read_dts[stream] == (int64_t)AV_NOPTS_VALUE)
            return 0;

        a1 = abcdk_ffeditor_ts2sec(ctx, stream, ctx->read_dts_first[stream]);
        a2 = abcdk_ffeditor_ts2sec(ctx, stream, ctx->read_dts[stream]) + (double)ctx->cfg.read_delay_max;
        a = (a2 - a1) - (a1 - a1);
        a /= ctx->cfg.read_speed; // 倍速。
        b = (double)(_abcdk_ffeditor_clock() - ctx->read_start[stream]) / 1000000.;

        block = (a >= b ? 0 : 1);

        //     abcdk_trace_printf(LOG_DEBUG,"stream(%d),flag(%d),a1(%.3f),a2(%.3f),a(%.3f),b(%.3f),block(%d)\n",stream,flag,a1,a2, a, b,block);
    }
    else
    {
        if (ctx->cfg.read_flush)
            return 1;

        /*如果是无效的DTS，直接返回1。*/
        if (ctx->read_dts[stream] == (int64_t)AV_NOPTS_VALUE)
            return 1;

        a1 = abcdk_ffeditor_ts2sec(ctx, stream, ctx->read_dts_first[stream]);
        a2 = abcdk_ffeditor_ts2sec(ctx, stream, ctx->read_dts[stream]);
        a = (a2 - a1) - (a1 - a1);
        a /= ctx->cfg.read_speed; // 倍速。
        b = (double)(_abcdk_ffeditor_clock() - ctx->read_start[stream]) / 1000000.;

        block = (a > b ? 0 : 1);

        //       abcdk_trace_printf(LOG_DEBUG,"stream(%d),flag(%d),a1(%.3f),a2(%.3f),a(%.3f),b(%.3f),block(%d)\n",stream,flag,a1,a2, a, b,block);
    }

    return block;
}

static void _abcdk_ffeditor_read_delay(abcdk_ffeditor_t *ctx)
{
    AVStream *vs_p = NULL;
    int64_t start_time = 0;
    int stream_idx = 0;
    int block = 0;

    assert(ctx != NULL);

next_delay:

    /*如果已经超时，则直接返回。*/
    if (_abcdk_ffeditor_interrupt_cb(ctx) != 0)
        return;

    for (int i = 0; i < abcdk_ffeditor_streams(ctx); i++)
    {
        vs_p = abcdk_ffeditor_streamptr(ctx, i);

        start_time = vs_p->start_time;
        stream_idx = vs_p->index;

        /*检测延时是否已经满足。*/
        block = !_abcdk_ffeditor_read_delay_check(ctx, stream_idx, 0);

        /*以最慢流的为基准。*/
        if (block)
            break;
    }

    if (block)
    {
        usleep(5000); // 200fps
        goto next_delay;
    }

    return;
}

int abcdk_ffeditor_read_packet(abcdk_ffeditor_t *ctx, AVPacket *pkt, int stream)
{
    AVStream *vs_p = NULL;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    AVCodecContext *codecpar = NULL;
#else
    AVCodecParameters *codecpar = NULL;
#endif
    int obsolete = 0;
    int chk;

    assert(ctx != NULL && pkt != NULL);

next_packet:

    /*Reset.*/
    obsolete = 0;

    /*等待下一帧时间到达。*/
    _abcdk_ffeditor_read_delay(ctx);

    chk = abcdk_avformat_input_read(ctx->avctx, pkt, AVMEDIA_TYPE_NB);
    if (chk < 0)
        return -1;

    vs_p = ctx->avctx->streams[pkt->stream_index];

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    codecpar = vs_p->codec;
#else
    codecpar = vs_p->codecpar;
#endif

    /*更新最近包时间，不然会超时。*/
    ctx->last_packet_time = _abcdk_ffeditor_clock();

    /*记录KEY帧和帧分组时间。*/
    if ((pkt->flags & AV_PKT_FLAG_KEY) || (codecpar->codec_type != AVMEDIA_TYPE_VIDEO))
        ctx->read_key_ns[pkt->stream_index] = ctx->read_gop_ns[pkt->stream_index] = _abcdk_ffeditor_clock();

    /*记录有效的DTS，并记录开始读取时间(用于记算拉流延时)。*/
    if (pkt->dts != (int64_t)AV_NOPTS_VALUE)
    {
        /*记录当前DTS。*/
        ctx->read_dts[pkt->stream_index] = pkt->dts;

        /*
         * 满足以下两个条件时，需要更新时间轴开始时间。
         * 1：开始时间无效时。
         * 2：时间轴重置。
         */
        if (ctx->read_dts_first[pkt->stream_index] == (int64_t)AV_NOPTS_VALUE ||
            ctx->read_dts_first[pkt->stream_index] > ctx->read_dts[pkt->stream_index])
        {
            ctx->read_dts_first[pkt->stream_index] = ctx->read_dts[pkt->stream_index];
            ctx->read_start[pkt->stream_index] = _abcdk_ffeditor_clock();
        }
    }

    /* 如果指定了流索引，这里筛一下。*/
    if (stream >= 0)
    {
        if (pkt->stream_index != stream)
            goto next_packet;
    }

    /*检测延时是否超过阈值。*/
    if (_abcdk_ffeditor_read_delay_check(ctx, pkt->stream_index, 1))
        ctx->read_gop_ns[pkt->stream_index] = 0;

    /*可能已经不在同一个GOP中。*/
    if (ctx->read_key_ns[pkt->stream_index] != ctx->read_gop_ns[pkt->stream_index])
        obsolete = 1;

    /*按需丢弃延时过多的帧，以便减少延时。*/
    if (obsolete)
    {
        abcdk_trace_printf(LOG_WARNING, TT("拉流延时超过设定阈值(delay_max=%.3f)，丢弃此数据包(index=%d,dts=%.3f,pts=%.3f)。"),
                           ctx->cfg.read_delay_max, pkt->stream_index,
                           abcdk_ffeditor_ts2sec(ctx, pkt->stream_index, pkt->dts),
                           abcdk_ffeditor_ts2sec(ctx, pkt->stream_index, pkt->pts));

        goto next_packet;
    }

    /*过滤器或其它处理。*/
    if (ctx->input_mp4_h264[pkt->stream_index] || ctx->input_mp4_h265[pkt->stream_index])
    {
        if (ctx->cfg.bit_stream_filter)
        {
            chk = abcdk_avformat_input_filter(ctx->avctx, pkt, &ctx->vs_filter[pkt->stream_index]);
            if (chk < 0)
                return -1;
        }
    }
    else if (ctx->input_mp4_mpeg4[pkt->stream_index])
    {
        /*fix me.*/;
    }

    return pkt->stream_index;
}

int abcdk_ffeditor_read_frame(abcdk_ffeditor_t *ctx, AVFrame *frame, int stream)
{
    AVCodecContext *codec_ctx_p;
    AVPacket *pkt_p;
    int chk = -1;

    assert(ctx != NULL && frame != NULL);

    av_frame_unref(frame);

next_packet:

    /*清空数输缓存。*/
    av_packet_unref(&ctx->read_pkt);

    if (!ctx->read_eof)
    {
        ctx->read_idx = abcdk_ffeditor_read_packet(ctx, &ctx->read_pkt, stream);
        if (ctx->read_idx < 0)
        {
            /*标记读到末尾或断开。*/
            ctx->read_eof = 1;

            /*如果未指定流，则从0号索引开始遍历延时解码帧。*/
            if (stream >= 0)
                ctx->read_idx = stream;
            else
                ctx->read_idx = 0;

            goto next_packet;
        }

        chk = _abcdk_ffeditor_capture_codec_init(ctx, ctx->read_idx);
        if (chk < 0)
            return -1;
    }

    codec_ctx_p = ctx->codec_ctx[ctx->read_idx];
    if (!codec_ctx_p)
    {
        /*
         * 以下两种情况直接返回失败。
         *
         * 1：流尚未结束。
         * 2：指定流索引。
         */
        if (!ctx->read_eof || stream >= 0)
            return -1;

        /*不能超过最大流索引。*/
        if (ctx->read_idx >= ABCDK_FFMPEG_MAX_STREAMS - 1)
            return -1;
        else
        {
            ctx->read_idx += 1; // next stream;
            goto next_packet;
        }
    }

    pkt_p = (ctx->read_eof ? NULL : &ctx->read_pkt);
    chk = abcdk_avcodec_decode(codec_ctx_p, frame, pkt_p);
    if (chk < 0)
        return -1;
    else
    {
        /*
         * 以下两种情况直接返回失败。
         *
         * 1：流尚未结束。
         * 2：指定流索引。
         */
        if (!ctx->read_eof || stream >= 0)
            return -1;

        /*不能超过最大流索引。*/
        if (ctx->read_idx >= ABCDK_FFMPEG_MAX_STREAMS - 1)
            return -1;
        else
        {
            ctx->read_idx += 1; // next stream;
            goto next_packet;
        }
    }

    return ctx->read_idx;
}

int _abcdk_ffeditor_writer_open_codec(abcdk_ffeditor_t *ctx, int stream, AVCodec *codec, const AVCodecContext *opt)
{
    AVCodecContext *ctx_p = NULL;
    AVDictionary *dict_p = NULL;
    AVStream *vs_p = NULL;
    int chk;

    if (stream >= ctx->avctx->nb_streams)
        return -1;

    if (stream >= ABCDK_FFMPEG_MAX_STREAMS)
        return -1;

    /* 如果已经打开，直接返回。*/
    if (ctx->codec_ctx[stream])
        return 0;

    if (!codec)
        return -1;

    vs_p = ctx->avctx->streams[stream];

    ctx_p = abcdk_avcodec_alloc(codec);
    if (!ctx_p)
        goto final_error;

    if (ctx_p->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        assert(opt->time_base.den > 0 && opt->time_base.num > 0);
        assert(opt->width > 0 && opt->height > 0);
        assert(opt->gop_size > 0);

        ctx_p->time_base = opt->time_base;
        ctx_p->framerate.den = opt->time_base.num;
        ctx_p->framerate.num = opt->time_base.den;
        ctx_p->pkt_timebase.num = opt->time_base.num;
        ctx_p->pkt_timebase.den = opt->time_base.den;

        ctx_p->width = opt->width;
        ctx_p->height = opt->height;
        ctx_p->gop_size = opt->gop_size;
        ctx_p->pix_fmt = opt->pix_fmt;

        if (ctx_p->pix_fmt == AV_PIX_FMT_NONE)
            ctx_p->pix_fmt = (ctx_p->codec->pix_fmts ? ctx_p->codec->pix_fmts[0] : AV_PIX_FMT_YUV420P);

        /*No b frame.*/
        ctx_p->max_b_frames = 0;
    }
    else if (ctx_p->codec_type == AVMEDIA_TYPE_AUDIO)
    {
        assert(opt->time_base.den > 0 && opt->time_base.num > 0);
        assert(opt->sample_rate > 0);
        assert(opt->channel_layout > 0);
        assert(opt->bit_rate > 0);
        assert(opt->frame_size > 0);

        ctx_p->time_base = opt->time_base;
        ctx_p->framerate.den = opt->time_base.num;
        ctx_p->framerate.num = opt->time_base.den;

        ctx_p->sample_rate = opt->sample_rate;
        ctx_p->channels = opt->channels;
        ctx_p->sample_fmt = opt->sample_fmt;
        ctx_p->channel_layout = opt->channel_layout;
        ctx_p->bit_rate = opt->bit_rate;
        ctx_p->frame_size = opt->frame_size;

        if (ctx_p->channels != av_get_channel_layout_nb_channels(opt->channel_layout))
            ctx_p->channels = av_get_channel_layout_nb_channels(opt->channel_layout);

        if (ctx_p->sample_fmt == AV_SAMPLE_FMT_NONE)
            ctx_p->sample_fmt = (ctx_p->codec->sample_fmts ? ctx_p->codec->sample_fmts[0] : AV_SAMPLE_FMT_FLTP);

        if (ctx_p->codec_id == AV_CODEC_ID_AAC)
            av_dict_set(&dict_p, "strict", "-2", 0);
    }
    else
    {
        goto final_error; // fix me.
    }

    /*如果流需要设置全局头部，则编码器需要知道这个请求。*/
    if (ctx->avctx->oformat->flags & AVFMT_GLOBALHEADER)
        ctx_p->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    chk = abcdk_avcodec_open(ctx_p, &dict_p);
    if (chk < 0)
        goto final_error;

    abcdk_avstream_parameters_from_context(vs_p, ctx_p);

    ctx->codec_ctx[stream] = ctx_p;
    ctx->codec_dict[stream] = dict_p;

    return 0;

final_error:

    abcdk_avcodec_free(&ctx_p);
    av_dict_free(&dict_p);

    return -1;
}

int abcdk_ffeditor_add_stream(abcdk_ffeditor_t *ctx, const AVCodecContext *opt, int have_codec)
{
    AVStream *vs = NULL;
    int chk = -1;

    assert(ctx != NULL && opt != NULL);

    if (ctx->avctx->nb_streams >= ABCDK_FFMPEG_MAX_STREAMS)
        return -2;

    vs = abcdk_avformat_output_stream(ctx->avctx, abcdk_avcodec_find2(opt->codec_id, 1));
    if (!vs)
        return -1;

    if (have_codec)
    {
        abcdk_avstream_parameters_from_context(vs, opt);

#if FF_API_LAVF_AVCTX
        /*如果流需要设置全局头部，则编码器需要知道这个请求。*/
        if (ctx->avctx->oformat->flags & AVFMT_GLOBALHEADER)
            vs->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
#endif //FF_API_LAVF_AVCTX

    }
    else
    {
        if (ctx->cfg.try_nvcodec)
        {
            /*NVCODEC硬件编码，必须用下面的写法，因为编码器可能未安装。*/
            if (opt->codec_id == AV_CODEC_ID_HEVC)
                chk = _abcdk_ffeditor_writer_open_codec(ctx, vs->index, abcdk_avcodec_find("hevc_nvenc", 1), opt);
            else if (opt->codec_id == AV_CODEC_ID_H264)
                chk = _abcdk_ffeditor_writer_open_codec(ctx, vs->index, abcdk_avcodec_find("h264_nvenc", 1), opt);
            else
                chk = -1;
        }

        /*如果硬件解码已经安装，到这里可能已经成功了。*/
        if (chk < 0)
            chk = _abcdk_ffeditor_writer_open_codec(ctx, vs->index, abcdk_avcodec_find2(opt->codec_id, 1), opt);

        if (chk < 0)
            return -1;
    }

    return vs->index;
}

int abcdk_ffeditor_write_header(abcdk_ffeditor_t *ctx, const AVDictionary *dict)
{
    int chk;

    assert(ctx != NULL);

    /*头部，写入一次就好。*/
    if (ctx->write_header_ok)
        return 0;

    /*复制外部的字典。*/
    if (dict)
        av_dict_copy(&ctx->dict, dict, 0);

    chk = abcdk_avformat_output_header(ctx->avctx, &ctx->dict);
    if (chk < 0)
        return -1;

    /*Set OK.*/
    ctx->write_header_ok = 1;

    return 0;
}

int abcdk_ffeditor_write_header_fmp4(abcdk_ffeditor_t *ctx)
{
    AVDictionary *dict = NULL;
    int chk;

    abcdk_avdict_make_fmp4(&dict);

    chk = abcdk_ffeditor_write_header(ctx, dict);

    av_dict_free(&dict);

    return chk;
}

int abcdk_ffeditor_write_trailer(abcdk_ffeditor_t *ctx)
{
    AVCodecContext *ctx_p = NULL;
    AVPacket pkt = {0};
    int idx;
    int chk;

    assert(ctx != NULL);

    /*写入头部后，才能写末尾。*/
    if (!ctx->write_header_ok)
        return -2;

    /*写入一次即可。*/
    if (ctx->write_trailer_ok)
        return 0;

    av_init_packet(&pkt);

    /* 写入所有延时编码数据包。*/
    for (int i = 0; i < ctx->avctx->nb_streams; i++)
    {
        idx = ctx->avctx->streams[i]->index;
        ctx_p = ctx->codec_ctx[idx];

        /*跳过使用外部编码器的流。*/
        if (!ctx_p)
            continue;

        /* 检查当前编码器是否支持延时编码。*/
        if (!(ctx_p->codec->capabilities & AV_CODEC_CAP_DELAY))
            continue;

        for (;;)
        {
            av_packet_unref(&pkt);
            chk = abcdk_avcodec_encode(ctx_p, &pkt, NULL);
            if (chk <= 0)
                break;

            pkt.stream_index = idx;

            chk = abcdk_ffeditor_write_packet(ctx, &pkt, NULL);
            if (chk < 0)
                goto final;
        }
    }

final:

    /*不要忘记。*/
    av_packet_unref(&pkt);

    /*标记已经写入末尾。*/
    ctx->write_trailer_ok = 1;

    chk = abcdk_avformat_output_trailer(ctx->avctx);
    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_ffeditor_write_packet(abcdk_ffeditor_t *ctx, AVPacket *pkt, AVRational *src_time_base)
{
    AVRational bq, cq;
    AVCodecContext *ctx_p = NULL;
    AVStream *vs_p = NULL;
    int chk;

    assert(ctx != NULL && pkt != NULL);
    assert(pkt->stream_index >= 0 && pkt->stream_index < ctx->avctx->nb_streams);

    /*写入头部后，才能写正文。*/
    if (!ctx->write_header_ok)
        return -2;

    ctx_p = ctx->codec_ctx[pkt->stream_index];
    vs_p = ctx->avctx->streams[pkt->stream_index];

    /*
     * 如果没有源时间基值，则使用内部时间基值。
     * 注：如果时间基值错误，编码数据在解码后无法正常播放，常见的现像是DTS、PTS、duration异常。
     */
    if (src_time_base)
        bq = *src_time_base;
    else
#if FF_API_LAVF_AVCTX
        bq = (ctx_p ? ctx_p->time_base : vs_p->codec->time_base);
#else //FF_API_LAVF_AVCTX
        bq = (ctx_p ? ctx_p->time_base : av_make_q(1, abcdk_avstream_fps(ctx->avctx, vs_p, 1)));
#endif //FF_API_LAVF_AVCTX

    cq = vs_p->time_base;

    /*修复错误的时长。*/
    if (pkt->duration == 0)
        pkt->duration = av_rescale_q(1, vs_p->time_base, AV_TIME_BASE_Q);

    /**/
    pkt->dts = av_rescale_q_rnd(pkt->dts, bq, cq, AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
    pkt->pts = av_rescale_q_rnd(pkt->pts, bq, cq, AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
    pkt->duration = av_rescale_q(pkt->duration, bq, cq);
    pkt->pos = -1;

    /* 确保DTS单调递增(简单修复) */
    if (pkt->dts <= ctx->write_last_dts[pkt->stream_index])
        pkt->dts = ctx->write_last_dts[pkt->stream_index] + 1;

    /* 记录最新的DTS*/
    ctx->write_last_dts[pkt->stream_index] = pkt->dts;

    /* 确保 DTS <= PTS */
    if (pkt->dts > pkt->pts)
        pkt->dts = pkt->pts;

    chk = abcdk_avformat_output_write(ctx->avctx, pkt, ctx->avctx->nb_streams, ctx->cfg.write_flush);

    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_ffeditor_write_packet2(abcdk_ffeditor_t *ctx, void *data, int size, int keyframe, int stream)
{
    AVPacket pkt = {0};
    AVStream *vs_p = NULL;
#if FF_API_LAVF_AVCTX
    AVCodecContext *codec_p;
#else //FF_API_LAVF_AVCTX
    AVCodecParameters *codec_p;
#endif //FF_API_LAVF_AVCTX
    int chk;

    assert(ctx != NULL && data != NULL && size > 0 && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    vs_p = ctx->avctx->streams[stream];

#if FF_API_LAVF_AVCTX
    codec_p = vs_p->codec;
#else //FF_API_LAVF_AVCTX
    codec_p = vs_p->codecpar;
#endif //FF_API_LAVF_AVCTX

    av_init_packet(&pkt);

    pkt.data = (uint8_t *)data;
    pkt.size = size;
    pkt.stream_index = stream;
    pkt.flags = (keyframe ? AV_PKT_FLAG_KEY : 0);

    if (codec_p->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        /*DTS值不能大于PTS。*/

        pkt.dts = ctx->ts_nums[pkt.stream_index][1]++;
        pkt.pts = ++ctx->ts_nums[pkt.stream_index][0];
    }
    else if (codec_p->codec_type == AVMEDIA_TYPE_AUDIO)
    {
        ABCDK_ASSERT(0, TT("fix me."));
    }

    chk = abcdk_ffeditor_write_packet(ctx, &pkt, NULL);
    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_ffeditor_write_frame(abcdk_ffeditor_t *ctx, AVFrame *frame, int stream)
{
    AVCodecContext *ctx_p = NULL;
    AVStream *vs_p = NULL;
    AVFrame *frame_cp = NULL;
    AVPacket pkt = {0};
    int chk = -1;

    assert(ctx != NULL && frame != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    ctx_p = ctx->codec_ctx[stream];
    vs_p = ctx->avctx->streams[stream];

    /*使用外部编器，不支持。*/
    if (!ctx_p)
        return -2;

    frame_cp = av_frame_clone(frame);
    if (!frame_cp)
        return -1;

    /*下面设置会使编码器重新计算播放时间。*/
    if (ctx_p->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        frame_cp->pts = ++ctx->ts_nums[stream][0];
    }
    else if (ctx_p->codec_type == AVMEDIA_TYPE_AUDIO)
    {
        frame_cp->pts = ctx->ts_nums[stream][0];
        ctx->ts_nums[stream][0] += frame_cp->nb_samples;
    }

    frame_cp->pkt_dts = (int64_t)AV_NOPTS_VALUE;
#if FF_API_PKT_PTS
    frame_cp->pkt_pts = (int64_t)AV_NOPTS_VALUE;
#endif //FF_API_PKT_PTS

    /*下面设置会使编码器自行决定帧类型。*/
    frame_cp->key_frame = 0;
    frame_cp->pict_type = AV_PICTURE_TYPE_NONE;

    av_init_packet(&pkt);
    chk = abcdk_avcodec_encode(ctx_p, &pkt, frame_cp);
    if (chk <= 0)
        goto final;

    pkt.stream_index = stream;
    chk = abcdk_ffeditor_write_packet(ctx, &pkt, NULL);

final:

    av_packet_unref(&pkt);
    av_frame_free(&frame_cp);

    return chk;
}

#pragma GCC diagnostic pop

#endif // AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

