/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/avformat.h"

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

void abcdk_avio_free(AVIOContext **ctx)
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

AVIOContext *abcdk_avio_alloc(int buf_blocks, int write_flag, void *opaque)
{
    int buf_size = 8 * 4096; /* 4k bytes 的倍数。 */
    void *buf = NULL;
    AVIOContext *ctx = NULL;

    if (buf_blocks > 0)
        buf_size = buf_blocks * 4096;

    buf = av_malloc(buf_size);
    if (!buf)
        goto final_error;

    ctx = avio_alloc_context((uint8_t *)buf, buf_size, write_flag, opaque, NULL, NULL, NULL);
    if (!ctx)
        goto final_error;

    return ctx;

final_error:

    av_freep(&buf);

    return NULL;
}

void abcdk_avformat_dump(AVFormatContext *ctx,int is_output)
{
    if (!ctx)
        return;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    av_dump_format(ctx, 0, ctx->filename, is_output);
#else
    av_dump_format(ctx, 0, ctx->url, is_output);
#endif
}

void abcdk_avformat_show_options(AVFormatContext *ctx)
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

void abcdk_avformat_free(AVFormatContext **ctx)
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
        abcdk_avio_free(&ctx_p->pb);
    else if(ctx_p->oformat && !(ctx_p->oformat->flags & AVFMT_NOFILE))
        avio_closep(&ctx_p->pb);

    if (ctx_p->iformat)
        avformat_close_input(&ctx_p);
    else
        avformat_free_context(ctx_p);
}

AVFormatContext *abcdk_avformat_input_open(const char *short_name, const char *filename,
                                           AVIOInterruptCB *interrupt, AVIOContext *io,
                                           AVDictionary **dict)
{
    AVInputFormat *fmt = NULL;
    AVFormatContext *ctx = NULL;
    int chk = -1;

    assert((filename != NULL && io == NULL) || (filename == NULL && io != NULL));
    assert(filename != NULL || (short_name != NULL && io != NULL));
    

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    av_register_all();
#endif
    avformat_network_init();
    avdevice_register_all();

    ctx = avformat_alloc_context();
    if (!ctx)
        return NULL;

    /*
     * 1: 如果不知道下面标志如何使用，一定不要附加这个标志。
     * 2: 如果附加此标志，会造成数据流开头的数据包丢失(N个)。
     * 3: 如果未附加此标志，网络流会产生不确定的延时(N毫秒~N秒)。
    */
    //ctx->flags |= AVFMT_FLAG_NOBUFFER;

    if (interrupt)
        ctx->interrupt_callback = *interrupt;

    if (io)
    {
        ctx->pb = io;
        ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
    }

    if (dict)
    {
        av_dict_set(dict, "scan_all_pmts", "1", AV_DICT_DONT_OVERWRITE);
        
        if(filename)
        {
            /* RTSP默认走TCP，可以减少丢包。*/
            if (strncmp(filename, "rtsp://", 7) == 0 || strncmp(filename, "rtsps://", 8) == 0)
            {
                av_dict_set(dict, "rtsp_transport", "tcp", AV_DICT_DONT_OVERWRITE);
            }
        }
    }

    fmt = (AVInputFormat *)av_find_input_format(short_name);
    chk = avformat_open_input(&ctx, filename, fmt, dict);

    if (chk != 0)
        abcdk_avformat_free(&ctx);

    return ctx;
}

int abcdk_avformat_input_probe(AVFormatContext *ctx, AVDictionary **dict)
{
    int chk = -1;

    assert(ctx != NULL);
    assert(ctx->iformat != NULL);

    return avformat_find_stream_info(ctx, dict);
}

int abcdk_avformat_input_read(AVFormatContext *ctx, AVPacket *pkt, enum AVMediaType only_type)
{
    int chk = -1;

    assert(ctx != NULL && pkt != NULL);

    for (;;)
    {
        av_packet_unref(pkt);
        chk = av_read_frame(ctx, pkt);
        if (chk < 0)
            return chk;

        if (only_type < AVMEDIA_TYPE_NB)
        {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
            if (ctx->streams[pkt->stream_index]->codec->codec_type == only_type)
#else
            if (ctx->streams[pkt->stream_index]->codecpar->codec_type == only_type)
#endif
                break;
        }
        else
        {
            break;
        }
    }

    return 0;
}

#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
int abcdk_avformat_input_filter(AVFormatContext *ctx, AVPacket *pkt, AVBSFContext **filter)
{
    const AVBitStreamFilter *bsf = NULL;
    AVBSFContext *filter_p = NULL;
    AVCodecParameters *codecpar = NULL;
    uint8_t *outbuf = NULL;
    int outbuf_size = 0;
    uint8_t *inbuf = NULL;
    int inbuf_size = 0;
    int chk = -1;

    assert(ctx != NULL && pkt != NULL);

    /*不能保存指针，什么也不做。*/
    if (!filter)
        return 0;

    filter_p = *filter;
    codecpar = ctx->streams[pkt->stream_index]->codecpar;

    /*如果过滤器未创建，则在内部创建。*/
    if (!filter_p)
    {
        if (codecpar->codec_id == AV_CODEC_ID_H264)
            bsf = av_bsf_get_by_name("h264_mp4toannexb");
        else if (codecpar->codec_id == AV_CODEC_ID_HEVC)
            bsf = av_bsf_get_by_name("hevc_mp4toannexb");
        // else if (codecpar->codec_id == AV_CODEC_ID_AAC)
        //     bsf = av_bsf_get_by_name("aac_adtstoasc");
        else
            return 0;

        if (!bsf)
            return 0;

        av_bsf_alloc(bsf, &filter_p);

        if (!filter_p)
            return -1;

        avcodec_parameters_copy(filter_p->par_in, codecpar);

        //av_opt_set_int(filter_p, "pps_id", -1, 0); // 只修复超出范围的ID。-1 表示保留原始ID，除非无效。

        av_bsf_init(filter_p);

        /*保存过滤器环境指针。*/
        *filter = filter_p;
    }

    av_bsf_send_packet(filter_p, pkt);
    av_packet_unref(pkt);
    chk = av_bsf_receive_packet(filter_p, pkt);
    assert(chk == 0);

    return 0;
}
#else
int abcdk_avformat_input_filter(AVFormatContext *ctx, AVPacket *pkt, AVBitStreamFilterContext **filter)
{
    AVBitStreamFilterContext *filter_p = NULL;
    AVCodecContext *codec_p = NULL;
    uint8_t *outbuf = NULL;
    int outbuf_size = 0;
    uint8_t *inbuf = NULL;
    int inbuf_size = 0;
    int chk = -1;

    assert(ctx != NULL && pkt != NULL);

    /*不能保存指针，什么也不做。*/
    if (!filter)
        return 0;

    filter_p = *filter;
    codec_p = ctx->streams[pkt->stream_index]->codec;

    /*如果过滤器未创建，则在内部创建。*/
    if (!filter_p)
    {
        if (codec_p->codec_id == AV_CODEC_ID_H264)
            filter_p = av_bitstream_filter_init("h264_mp4toannexb");
        else if (codec_p->codec_id == AV_CODEC_ID_HEVC)
            filter_p = av_bitstream_filter_init("hevc_mp4toannexb");
        // else if (codec_p->codec_id == AV_CODEC_ID_AAC)
        //     filter_p = av_bitstream_filter_init("aac_adtstoasc");
        else
            return 0;

        if (!filter_p)
            return -1;

        //av_opt_set_int(filter_p->priv_data, "aud", 2, 0); // 移除TYPE=9(AUD)

        /*保存过滤器环境指针。*/
        *filter = filter_p;
    }

    inbuf = pkt->data;
    inbuf_size = pkt->size;
    chk = av_bitstream_filter_filter(filter_p, codec_p, NULL, &outbuf, &outbuf_size, inbuf, inbuf_size, 0); // pkt->flags & AV_PKT_FLAG_KEY);
    if (chk < 0)
        return -1;

    /*
     * 1: 当返回值大于0时，需要释放旧的内存，绑定新的内存。
     * 2: 当返回值等于0时，未创建新的内存，但是数据起始地址可能有变化。
     */
    if (chk > 0)
    {
        av_buffer_unref(&pkt->buf);
        pkt->buf = av_buffer_create(outbuf, outbuf_size, NULL, NULL, 0);
    }

    pkt->data = outbuf;
    pkt->size = outbuf_size;

    return 0;
}
#endif

AVFormatContext *abcdk_avformat_output_open(const char *short_name, const char *filename, const char *mime_type,
                                            AVIOInterruptCB *interrupt, AVIOContext *io)
{
    AVInputFormat *fmt = NULL;
    AVFormatContext *ctx = NULL;
    int chk = -1;

    assert((filename != NULL && io == NULL) || (filename == NULL && io != NULL));
    assert(filename != NULL || (short_name != NULL && io != NULL));

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    av_register_all();
#endif
    avformat_network_init();
    avdevice_register_all();

    ctx = avformat_alloc_context();
    if (!ctx)
        return NULL;

    if (filename)
    {
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
        strncpy(ctx->filename, filename, sizeof(ctx->filename));
#else
        ctx->url = av_strdup(filename);
#endif

        if (abcdk_strncmp(filename, "rtsp://", 7, 0) == 0)
            ctx->oformat = av_guess_format("rtsp", NULL, NULL);
        else if (abcdk_strncmp(filename, "rtsps://", 8, 0) == 0)
            ctx->oformat = av_guess_format("rtsp", NULL, NULL);
        else if (abcdk_strncmp(filename, "rtmp://", 7, 0) == 0)
            ctx->oformat = av_guess_format("flv", NULL, NULL);
        else if (abcdk_strncmp(filename, "rtmps://", 8, 0) == 0)
            ctx->oformat = av_guess_format("flv", NULL, NULL);
    }

    if (!ctx->oformat)
        ctx->oformat = av_guess_format(short_name, filename, mime_type);

    if (!ctx->oformat)
        goto final_error;
    
#if 0
    av_dict_set(&ctx->metadata, "service", ABCDK, 0);
    av_dict_set(&ctx->metadata, "service_name", ABCDK, 0);
    av_dict_set(&ctx->metadata, "service_provider", ABCDK, 0);
    av_dict_set(&ctx->metadata, "artist", ABCDK, 0);
#endif

    if (interrupt)
        ctx->interrupt_callback = *interrupt;

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

AVStream *abcdk_avformat_output_stream(AVFormatContext *ctx, const AVCodec *codec)
{
    assert(ctx != NULL && codec != NULL);
    assert(ctx->oformat);

    return avformat_new_stream(ctx, codec);
}

AVStream *abcdk_avformat_output_stream2(AVFormatContext *ctx, const char *name)
{
    return abcdk_avformat_output_stream(ctx, abcdk_avcodec_find(name, 1));
}

AVStream *abcdk_avformat_output_stream3(AVFormatContext *ctx, enum AVCodecID id)
{
    return abcdk_avformat_output_stream(ctx, abcdk_avcodec_find2(id, 1));
}

int abcdk_avformat_output_header(AVFormatContext *ctx, AVDictionary **dict)
{
    const char *url_p;
    int chk = -1;

    assert(ctx != NULL);
    assert(ctx->oformat);

    if ((ctx->oformat->flags & AVFMT_NOFILE) || ctx->pb)
        chk = 0;
    else
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
        chk = avio_open(&ctx->pb, ctx->filename, AVIO_FLAG_WRITE);
#else
        chk = avio_open(&ctx->pb, ctx->url, AVIO_FLAG_WRITE);
#endif
    if (chk != 0)
        return -1;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 20, 100)
    url_p = ctx->filename;
#else   
    url_p = ctx->url;
#endif

    if (dict)
    {
        if(url_p)
        {
            if (abcdk_strncmp(url_p, "rtsp://", 7, 0) == 0 || abcdk_strncmp(url_p, "rtsps://", 8, 0) == 0)
            {
                av_dict_set(dict, "rtsp_transport", "tcp", AV_DICT_DONT_OVERWRITE);
                av_dict_set(dict, "max_interleave_delta", "0", AV_DICT_DONT_OVERWRITE);
            }
        }
    }

    chk = avformat_write_header(ctx, dict);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_avformat_output_write(AVFormatContext *ctx, AVPacket *pkt, int tracks, int flush)
{
    int chk;
    assert(ctx != NULL && pkt != NULL && tracks >= 1);

    chk = (tracks > 1 ? av_interleaved_write_frame(ctx, pkt) : av_write_frame(ctx, pkt));
    //chk = av_interleaved_write_frame(ctx, pkt);
    if(chk != 0)
        return -1;

    if(flush && ctx->pb)
       avio_flush(ctx->pb);

    return 0;
}

int abcdk_avformat_output_trailer(AVFormatContext *ctx)
{
    assert(ctx != NULL);

    return av_write_trailer(ctx);
}

int abcdk_avstream_parameters_from_context(AVStream *vs, const AVCodecContext *ctx)
{
    assert(vs != NULL && ctx != NULL);

    /*如果是编码，帧率也一并复制。*/
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)
    if (av_codec_is_encoder(vs->codec->codec))
#else 
    if (av_codec_is_encoder(ctx->codec))
#endif 
    {
        vs->time_base = ctx->time_base;
        vs->avg_frame_rate = vs->r_frame_rate = ctx->framerate;//av_make_q(ctx->time_base.den, ctx->time_base.num);
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)
        vs->codec->time_base = ctx->time_base;
        vs->codec->framerate = ctx->framerate;
#endif
    }

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 35, 100)
    avcodec_parameters_from_context(vs->codecpar, ctx);
#endif 

    /*下面的也要复制，因为一些定制的ffmpeg未完成启用新的参数。*/
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)
    vs->codec->codec_type = ctx->codec_type;
    vs->codec->codec_id = ctx->codec_id;
    vs->codec->codec_tag = ctx->codec_tag;
    vs->codec->bit_rate = ctx->bit_rate;
    vs->codec->bits_per_coded_sample = ctx->bits_per_coded_sample;
    vs->codec->bits_per_raw_sample = ctx->bits_per_raw_sample;
    vs->codec->profile = ctx->profile;
    vs->codec->level = ctx->level;
    vs->codec->flags = ctx->flags;

    switch (ctx->codec_type)
    {
    case AVMEDIA_TYPE_VIDEO:
        vs->codec->pix_fmt = ctx->pix_fmt;
        vs->codec->width = ctx->width;
        vs->codec->height = ctx->height;
        vs->codec->gop_size = ctx->gop_size;
        vs->codec->field_order = ctx->field_order;
        vs->codec->color_range = ctx->color_range;
        vs->codec->color_primaries = ctx->color_primaries;
        vs->codec->color_trc = ctx->color_trc;
        vs->codec->colorspace = ctx->colorspace;
        vs->codec->chroma_sample_location = ctx->chroma_sample_location;
        vs->codec->sample_aspect_ratio = ctx->sample_aspect_ratio;
        vs->codec->has_b_frames = ctx->has_b_frames;
        break;
    case AVMEDIA_TYPE_AUDIO:
        vs->codec->sample_fmt = ctx->sample_fmt;
        vs->codec->channel_layout = ctx->channel_layout;
        vs->codec->channels = ctx->channels;
        vs->codec->sample_rate = ctx->sample_rate;
        vs->codec->block_align = ctx->block_align;
        vs->codec->frame_size = ctx->frame_size;
        vs->codec->initial_padding = ctx->initial_padding;
        vs->codec->seek_preroll = ctx->seek_preroll;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        vs->codec->width = ctx->width;
        vs->codec->height = ctx->height;
        break;
    default:
        break;
    }

    if (ctx->extradata)
    {
        vs->codec->extradata_size = 0;
        av_free(vs->codec->extradata);

        vs->codec->extradata = (uint8_t *)av_mallocz(ctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (vs->codec->extradata)
        {
            memcpy(vs->codec->extradata, ctx->extradata, ctx->extradata_size);
            vs->codec->extradata_size = ctx->extradata_size;
        }
        else
        {
            av_log(NULL, AV_LOG_INFO, "@av_mallocz ENOMEM!");
        }
    }
#endif //#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)

    return 0;
}

int abcdk_avstream_parameters_to_context(AVCodecContext *ctx, const AVStream *vs)
{
    assert(vs != NULL && ctx != NULL);

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 35, 100)
    avcodec_parameters_to_context(ctx, vs->codecpar);
#endif

    /*下面的也要复制，因为一些定制的ffmpeg未完成启用新的参数。*/
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)
    ctx->time_base = vs->codec->time_base;
    ctx->framerate = vs->codec->framerate;
    ctx->codec_type = vs->codec->codec_type;
    ctx->codec_id = vs->codec->codec_id;
    ctx->codec_tag = vs->codec->codec_tag;
    ctx->bit_rate = vs->codec->bit_rate;
    ctx->bits_per_coded_sample = vs->codec->bits_per_coded_sample;
    ctx->bits_per_raw_sample = vs->codec->bits_per_raw_sample;
    ctx->profile = vs->codec->profile;
    ctx->level = vs->codec->level;
    ctx->flags = vs->codec->flags;

    switch (vs->codec->codec_type)
    {
    case AVMEDIA_TYPE_VIDEO:
        ctx->pix_fmt = vs->codec->pix_fmt;
        ctx->width = vs->codec->width;
        ctx->height = vs->codec->height;
        ctx->field_order = vs->codec->field_order;
        ctx->color_range = vs->codec->color_range;
        ctx->color_primaries = vs->codec->color_primaries;
        ctx->color_trc = vs->codec->color_trc;
        ctx->colorspace = vs->codec->colorspace;
        ctx->chroma_sample_location = vs->codec->chroma_sample_location;
        ctx->sample_aspect_ratio = vs->codec->sample_aspect_ratio;
        ctx->has_b_frames = vs->codec->has_b_frames;
        break;
    case AVMEDIA_TYPE_AUDIO:
        ctx->sample_fmt = vs->codec->sample_fmt;
        ctx->channel_layout = vs->codec->channel_layout;
        ctx->channels = vs->codec->channels;
        ctx->sample_rate = vs->codec->sample_rate;
        ctx->block_align = vs->codec->block_align;
        ctx->frame_size = vs->codec->frame_size;
        ctx->delay = ctx->initial_padding = vs->codec->initial_padding;
        ctx->seek_preroll = vs->codec->seek_preroll;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        ctx->width = vs->codec->width;
        ctx->height = vs->codec->height;
        break;
    default:
        break;
    }

    if (vs->codec->extradata)
    {
        ctx->extradata_size = 0;
        av_free(ctx->extradata);

        ctx->extradata = (uint8_t *)av_mallocz(vs->codec->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (ctx->extradata)
        {
            memcpy(ctx->extradata, vs->codec->extradata, vs->codec->extradata_size);
            ctx->extradata_size = vs->codec->extradata_size;
        }
        else
        {
            av_log(NULL, AV_LOG_INFO, "@av_mallocz ENOMEM!");
        }
    }
#endif //#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)

    return 0;
}

/*-------------Copy from OpenCV----begin------------------*/

double abcdk_avstream_timebase_q2d(AVFormatContext *ctx,AVStream *vs,double xspeed)
{
    assert(ctx != NULL && vs != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);
    
    return abcdk_avmatch_r2d(vs->time_base,xspeed);
}

double abcdk_avstream_duration(AVFormatContext *ctx, AVStream *vs,double xspeed)
{
    double sec = 0.0;

    assert(ctx != NULL && vs != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);

    if (vs->duration > 0)
        sec = (double)vs->duration * abcdk_avmatch_r2d(vs->time_base,xspeed);

    return sec;
}

double abcdk_avstream_fps(AVFormatContext *ctx, AVStream *vs,double xspeed)
{
    double fps = -0.001;

    assert(ctx != NULL && vs != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);
    
#define ABCDK_AVSTREAM_EPS_ZERO 0.000025

    if (fps < ABCDK_AVSTREAM_EPS_ZERO)
        fps = abcdk_avmatch_r2d(vs->r_frame_rate,xspeed);
    if (fps < ABCDK_AVSTREAM_EPS_ZERO)
        fps = abcdk_avmatch_r2d(vs->avg_frame_rate,xspeed);
    if (fps < ABCDK_AVSTREAM_EPS_ZERO)
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)
        fps = 1.0 / abcdk_avmatch_r2d(vs->codec->time_base,xspeed);
#else 
        fps = 1.0 / abcdk_avmatch_r2d(vs->time_base,xspeed);
#endif 

    return ABCDK_CLAMP(fps,(double)1.0,(double)999999999.0);
}

double abcdk_avstream_ts2sec(AVFormatContext *ctx, AVStream *vs, int64_t ts, double xspeed)
{
    double sec = -0.000001;

    assert(ctx != NULL && vs != NULL && xspeed >= 0.001);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);

    sec = (double)(ts - vs->start_time) * abcdk_avmatch_r2d(vs->time_base, 1.0 / xspeed);

    return sec;
}

int64_t abcdk_avstream_ts2num(AVFormatContext *ctx, AVStream *vs, int64_t ts, double xspeed)
{
    int64_t frame_nb = -1;
    double sec = -0.000001;

    sec = abcdk_avstream_ts2sec(ctx, vs, ts, xspeed);
    if (sec >= 0.0)
        frame_nb = (int64_t)(abcdk_avstream_fps(ctx, vs, xspeed) * sec + 0.5);

    return frame_nb;
}

/*-------------Copy from OpenCV----end------------------*/

int abcdk_avstream_width(AVFormatContext *ctx, AVStream *vs)
{

    assert(ctx != NULL && vs != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    return vs->codec->width;
#else 
    return vs->codecpar->width;
#endif
}

int abcdk_avstream_height(AVFormatContext *ctx, AVStream *vs)
{

    assert(ctx != NULL && vs != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    return vs->codec->height;
#else 
    return vs->codecpar->height;
#endif
}

AVStream *abcdk_avstream_find(AVFormatContext *ctx,enum AVMediaType type)
{
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    AVCodecContext *codecpar = NULL;
#else 
    AVCodecParameters *codecpar = NULL;
#endif
    AVStream *vs_p;

    assert(ctx != NULL && type > AVMEDIA_TYPE_UNKNOWN && type < AVMEDIA_TYPE_NB);

    for (int i = 0; i < ctx->nb_streams; i++)
    {
        vs_p = ctx->streams[i];
    #if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
        codecpar = vs_p->codec;
    #else 
        codecpar = vs_p->codecpar;
    #endif

        if(codecpar->codec_type == type)
            return vs_p;
    }

    return NULL;
}

#pragma GCC diagnostic pop

#endif // AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H
