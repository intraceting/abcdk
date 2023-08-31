/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/ffmpeg/ffmpeg.h"

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

/** 最大支持16个。*/
#define ABCDK_FFMPEG_MAX_STREAMS     16

/** 视频对象。*/
typedef struct _abcdk_ffmpeg
{
    /** 编/解码器。*/
    AVCodecContext *codec_ctx[ABCDK_FFMPEG_MAX_STREAMS];

    /** 编/解码器字典。*/
    AVDictionary *codec_dict[ABCDK_FFMPEG_MAX_STREAMS];

    /** 数据包过滤器。*/
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58,20,100)
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

    /** 读数据包计数器。*/
    uint64_t read_pkt_count[ABCDK_FFMPEG_MAX_STREAMS];

    /** 读开始时间(系统时间，微秒)*/
    uint64_t read_start[ABCDK_FFMPEG_MAX_STREAMS];

    /** 当前DTS。*/
    int64_t read_dts[ABCDK_FFMPEG_MAX_STREAMS];

    /** 流容器。*/
    AVFormatContext *avctx;

    /** 流字典。*/
    AVDictionary *dict;

    /** 是否尝试NVCODEC编解码器。*/
    int try_nvcodec;

    /** 超时(秒)。*/
    int64_t timeout;

    /** 最近活动包时间(秒)。*/
    int64_t last_packet_time;

    /** 读缓存包.*/
    AVPacket read_pkt;

    /** 是否已经结束。*/
    int read_eof;

    /**
     * TS编号。
     * 
     * 0: PTS
     * 1: DTS
    */
    int64_t ts_nums[ABCDK_FFMPEG_MAX_STREAMS][2];

    /** 是否已经写入文件头部。*/
    int write_header_ok;

} abcdk_ffmpeg_t;


int64_t _abcdk_ffmpeg_clock()
{
    abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6);
}

void abcdk_ffmpeg_destroy(abcdk_ffmpeg_t **ctx)
{
    abcdk_ffmpeg_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    for (int i = 0; i < ABCDK_FFMPEG_MAX_STREAMS; i++)
    {
        if(ctx_p->codec_ctx[i])
            abcdk_avcodec_free(&ctx_p->codec_ctx[i]);

        if(ctx_p->codec_dict[i])
            av_dict_free(&ctx_p->codec_dict[i]);

        if(ctx_p->vs_filter[i])
        {
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58,20,100)
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

abcdk_ffmpeg_t *abcdk_ffmpeg_alloc()
{
    abcdk_ffmpeg_t *ctx = NULL;
    int chk;

    ctx = abcdk_heap_alloc(sizeof(abcdk_ffmpeg_t));
    if(!ctx)
        return NULL;

    
    return ctx;

final_error:

    abcdk_ffmpeg_destroy(&ctx);

    return NULL;
}

AVFormatContext *abcdk_ffmpeg_ctxptr(abcdk_ffmpeg_t *ctx)
{
    assert(ctx != NULL);

    return ctx->avctx;
}

int abcdk_ffmpeg_streams(abcdk_ffmpeg_t *ctx)
{
    assert(ctx != NULL);

    return ctx->avctx->nb_streams;
}

AVStream *abcdk_ffmpeg_streamptr(abcdk_ffmpeg_t *ctx,int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    return ctx->avctx->streams[stream];
}

double abcdk_ffmpeg_duration(abcdk_ffmpeg_t *ctx,int stream,double xspeed)
{
    assert(ctx != NULL && stream >= 0 && xspeed > 0.001);
    assert(stream < ctx->avctx->nb_streams);

    return abcdk_avstream_duration(ctx->avctx,ctx->avctx->streams[stream],xspeed);
}

double abcdk_ffmpeg_fps(abcdk_ffmpeg_t *ctx,int stream,double xspeed)
{
    assert(ctx != NULL && stream >= 0 && xspeed > 0.001);
    assert(stream < ctx->avctx->nb_streams);

    return abcdk_avstream_fps(ctx->avctx,ctx->avctx->streams[stream],xspeed);
}

double abcdk_ffmpeg_ts2sec(abcdk_ffmpeg_t *ctx,int stream, int64_t ts,double xspeed)
{
    assert(ctx != NULL && stream >= 0 && xspeed > 0.001);
    assert(stream < ctx->avctx->nb_streams);

    return abcdk_avstream_ts2sec(ctx->avctx,ctx->avctx->streams[stream],ts,xspeed);
}

int64_t abcdk_ffmpeg_ts2num(abcdk_ffmpeg_t *ctx,int stream, int64_t ts,double xspeed)
{
    assert(ctx != NULL && stream >= 0 && xspeed > 0.001);
    assert(stream < ctx->avctx->nb_streams);

    return abcdk_avstream_ts2num(ctx->avctx,ctx->avctx->streams[stream],ts,xspeed);
}

int abcdk_ffmpeg_width(abcdk_ffmpeg_t *ctx,int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    return abcdk_avstream_width(ctx->avctx,ctx->avctx->streams[stream]);
}

int abcdk_ffmpeg_height(abcdk_ffmpeg_t *ctx,int stream)
{
    assert(ctx != NULL && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    return abcdk_avstream_height(ctx->avctx,ctx->avctx->streams[stream]);
}

int _abcdk_ffmpeg_capture_interrupt_cb(void *args)
{
    abcdk_ffmpeg_t *ctx = (abcdk_ffmpeg_t *)args;
    uint64_t cur_time = _abcdk_ffmpeg_clock();

    if (ctx->timeout > 0)
    {
        /* 如果超时，返回失败。*/
        if ((cur_time - ctx->last_packet_time)/1000000 >= ctx->timeout)
            return -1;
    }

    return 0;
}

int _abcdk_ffmpeg_init_capture(abcdk_ffmpeg_t *ctx, const char *short_name, const char *url,AVIOContext *io, abcdk_option_t *opt)
{
    int is_mp4_file = 0;
    int chk;

    if(opt)
    {
        ctx->try_nvcodec = abcdk_option_get_int(opt,"--try-nvcodec",0,0);
        ctx->timeout = abcdk_option_get_int(opt,"--timeout",0,30);
    }

    ctx->last_packet_time = _abcdk_ffmpeg_clock();

    AVIOInterruptCB cb;
    cb.callback = _abcdk_ffmpeg_capture_interrupt_cb;
    cb.opaque = ctx;

    av_init_packet(&ctx->read_pkt);

    ctx->avctx = abcdk_avformat_input_open(short_name,url,&cb,io,&ctx->dict);
    if(!ctx->avctx)
        return -1;

    chk = abcdk_avformat_input_probe(ctx->avctx, NULL);
    if (chk < 0)
        return -1;

    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "QuickTime / MOV") == 0 ? 0x1 : 0);
    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "FLV (Flash Video)") == 0 ? 0x2 : 0);
    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "Matroska / WebM") == 0 ? 0x4 : 0);

    for(int i = 0;i<ctx->avctx->nb_streams;i++)
    {
        ctx->input_mp4_h264[i] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_H264 && is_mp4_file);
        ctx->input_mp4_h265[i] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_HEVC && is_mp4_file);
        ctx->input_mp4_mpeg4[i] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_MPEG4 && is_mp4_file);

        /*记录每个流的开始读取时间。*/
        ctx->read_start[i] = _abcdk_ffmpeg_clock();
    }

    return 0;
}

int _abcdk_ffmpeg_init_writer(abcdk_ffmpeg_t *ctx,const char *short_name, const char *url,AVIOContext *io, abcdk_option_t *opt)
{
    const char *mime_type_p = NULL;

    if(opt)
    {
        ctx->try_nvcodec = abcdk_option_get_int(opt,"--try-nvcodec",0,0);
        mime_type_p = abcdk_option_get(opt,"--mime-type",0,NULL);
    }

    ctx->avctx = abcdk_avformat_output_open(short_name, url, mime_type_p, NULL, io);
    if(!ctx->avctx)
        return -1;

    return 0;
}

abcdk_ffmpeg_t *abcdk_ffmpeg_open(int writer, const char *short_name, const char *url, AVIOContext *io, abcdk_option_t *opt)
{
    abcdk_ffmpeg_t *ctx = NULL;
    int chk;

    assert(url != NULL);

    ctx= abcdk_ffmpeg_alloc();
    if(!ctx)
        return NULL;

    if(writer)
        chk = _abcdk_ffmpeg_init_writer(ctx,short_name,url,io,opt);
    else 
        chk = _abcdk_ffmpeg_init_capture(ctx,short_name,url,io,opt);

    if (chk == 0)
        return ctx;

    abcdk_ffmpeg_destroy(&ctx);

    return NULL;
}

abcdk_ffmpeg_t *abcdk_ffmpeg_open_capture(const char *short_name, const char *url,int timeout)
{
    abcdk_ffmpeg_t *ctx = NULL;
    abcdk_option_t *opt = NULL;
    
    opt = abcdk_option_alloc("--");
    if(!opt)
        return NULL;

    abcdk_option_fset(opt,"--timeout","%d",timeout);

    ctx = abcdk_ffmpeg_open(0,short_name,url,NULL,opt);

    /*free option.*/
    abcdk_option_free(&opt);

    return ctx;
}

abcdk_ffmpeg_t *abcdk_ffmpeg_open_writer(const char*short_name,const char *url,const char *mime_type)
{
    abcdk_ffmpeg_t *ctx = NULL;
    abcdk_option_t *opt = NULL;
    
    opt = abcdk_option_alloc("--");
    if(!opt)
        return NULL;

    abcdk_option_fset(opt,"--mime-type","%s",mime_type);

    ctx = abcdk_ffmpeg_open(1,short_name,url,NULL,opt);

    /*free option.*/
    abcdk_option_free(&opt);

    return ctx;
}

int _abcdk_ffmpeg_capture_open_codec(abcdk_ffmpeg_t *ctx, int stream, AVCodec *codec)
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
    if(ctx->codec_ctx[stream])
        return 0;

    if(!codec)
        return -1;

    vs_p = ctx->avctx->streams[stream];

    ctx_p = abcdk_avcodec_alloc(codec);
    if(!ctx_p)
        goto final_error;
    
    abcdk_avstream_parameters_to_context(ctx_p, vs_p);

    ctx_p->pkt_timebase.num = vs_p->time_base.num;
    ctx_p->pkt_timebase.den = vs_p->time_base.den;
    
    chk = abcdk_avcodec_open(ctx_p, &dict_p);
    if(chk <0 )
        goto final_error;

    ctx->codec_ctx[stream] = ctx_p;
    ctx->codec_dict[stream] = dict_p;

    return 0;

final_error:

    abcdk_avcodec_free(&ctx_p);
    av_dict_free(&dict_p);

    return -1;
}

int _abcdk_ffmpeg_capture_codec_init(abcdk_ffmpeg_t *ctx, int stream)
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

    if(ctx->try_nvcodec)
    {
        /*NVCODEC硬件解码，必须用下面的写法，因为解码器可能未安装。*/
        if (codecpar->codec_id == AV_CODEC_ID_HEVC)
            _abcdk_ffmpeg_capture_open_codec(ctx, stream, abcdk_avcodec_find("hevc_cuvid", 0));
        else if (codecpar->codec_id == AV_CODEC_ID_H264)
            _abcdk_ffmpeg_capture_open_codec(ctx, stream, abcdk_avcodec_find("h264_cuvid", 0));
    }

    /*如果硬件解码已经安装，到这里可能已经成功了。*/
    codec_ctx_p = ctx->codec_ctx[stream];
    if (codec_ctx_p)
        return 0;

    chk = _abcdk_ffmpeg_capture_open_codec(ctx, stream, abcdk_avcodec_find2(codecpar->codec_id, 0));
    if (chk < 0)
        return -1;

    return 0;
}

void abcdk_ffmpeg_read_delay(abcdk_ffmpeg_t *ctx, double xspeed)
{
    AVStream * vs_p = NULL;
    int64_t start_time = 0;
    int block = 0;

    assert(ctx != NULL);

next_delay:

    for (int i = 0; i < abcdk_ffmpeg_streams(ctx); i++)
    {
        vs_p = abcdk_ffmpeg_streamptr(ctx,i);
        start_time = vs_p->start_time;

        /*流的起始值可能不为0，这里要加上，内部计算时会减掉。*/
        double a = abcdk_ffmpeg_ts2sec(ctx, i , ctx->read_dts[i] + start_time , xspeed);
        double b = (double)(_abcdk_ffmpeg_clock() - ctx->read_start[i]) / 1000000;

        //printf("%.3f == %.3f\n", a, b);
        
        /*以最慢的为准。*/
        if(block = (a > b ? 1 : 0))
            break;
    }

    if (block)
    {
        usleep(1000);
        goto next_delay;
    }
}

int abcdk_ffmpeg_read(abcdk_ffmpeg_t *ctx, AVPacket *pkt, int stream)
{
    uint8_t *extdata_p = NULL;
    int extsize = 0;
    int oldsize = 0;
    int chk;

    assert(ctx != NULL && pkt != NULL);

next_packet:

    chk = abcdk_avformat_input_read(ctx->avctx, pkt, AVMEDIA_TYPE_NB);
    if (chk < 0)
        return -1;

    /*读数据包 +1。*/
    ctx->read_pkt_count[pkt->stream_index] += 1;

    /*记录当前DTS。*/
    ctx->read_dts[pkt->stream_index] = pkt->dts;

    /* 更新最近包时间，不然会超时。*/
    ctx->last_packet_time = _abcdk_ffmpeg_clock();

    /* 如果指定了流索引，这里筛一下。*/
    if (stream >= 0)
    {
        if (pkt->stream_index != stream)
            goto next_packet;
    }

    if (ctx->input_mp4_mpeg4[pkt->stream_index])
    {
        /*mp4格式的mpeg码流需要特殊处理一下。*/

        // if (ctx->read_pkt_count[pkt->stream_index] == 1)
        // {
        //     extdata_p = ctx->avctx->streams[pkt->stream_index]->codec->extradata;
        //     extsize = ctx->avctx->streams[pkt->stream_index]->codec->extradata_size;

        //     if (extsize > 0)
        //     {
        //         /*记录现有数据长度。*/
        //         oldsize = pkt->size;
        //         /*mpeg全局头部有三个字节(0x00 0x00 0x01)的启起码，因此要减去。*/
        //         av_grow_packet(pkt, extsize - 3);
        //         /*把现有数据向后移动。*/
        //         memmove(pkt->data + (extsize - 3), pkt->data, oldsize);
        //         /*复制全局数据到开头。*/
        //         memcpy(pkt->data, extdata_p + 3,extsize - 3);
        //     }
        // }
    }
    else if (ctx->input_mp4_h264[pkt->stream_index] || ctx->input_mp4_h265[pkt->stream_index])
    {
        /*只有mp4格式的h264、h265才需要执行下面的过滤器。*/

        chk = abcdk_avformat_input_filter(ctx->avctx, pkt, &ctx->vs_filter[pkt->stream_index]);
        if (chk < 0)
            return -1;
    }

    return pkt->stream_index;
}

int abcdk_ffmpeg_read2(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream)
{
    AVCodecContext *codec_ctx_p;
    int chk = -1;

    assert(ctx != NULL && frame != NULL);

    av_frame_unref(frame);

next_packet:

    /*清空数输缓存。*/
    av_packet_unref(&ctx->read_pkt);

    if (!ctx->read_eof)
    {
        chk = abcdk_ffmpeg_read(ctx, &ctx->read_pkt, stream);
        if (chk < 0)
            ctx->read_eof = 1;

        chk = _abcdk_ffmpeg_capture_codec_init(ctx, ctx->read_pkt.stream_index);
        if (chk < 0)
            return -1;
    }

    codec_ctx_p = ctx->codec_ctx[ctx->read_pkt.stream_index];
    if (!codec_ctx_p)
        return -1;

    chk = abcdk_avcodec_decode(codec_ctx_p, frame, &ctx->read_pkt);
    if (chk < 0)
        return -1;
    if (chk == 0)
    {
        if(!ctx->read_eof)
            goto next_packet;
        else 
            return -1;
    }

    return ctx->read_pkt.stream_index;
}

int _abcdk_ffmpeg_writer_open_codec(abcdk_ffmpeg_t *ctx, int stream, AVCodec *codec,const AVCodecContext *opt)
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
    if(ctx->codec_ctx[stream])
        return 0;

    if(!codec)
        return -1;

    vs_p = ctx->avctx->streams[stream];

    ctx_p = abcdk_avcodec_alloc(codec);
    if(!ctx_p)
        goto final_error;

    if(ctx_p->codec_type == AVMEDIA_TYPE_VIDEO)
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

        /*像素格式必须在支持列表中。*/
        if (ctx_p->codec->pix_fmts)
        {
            int i = 0;
            for (; ctx_p->codec->pix_fmts[i] != AV_PIX_FMT_NONE; i++)
            {
                if (opt->pix_fmt == ctx_p->codec->pix_fmts[i])
                    break;
            }

            if(ctx_p->codec->pix_fmts[i] != AV_PIX_FMT_NONE)
                ctx_p->pix_fmt = ctx_p->codec->pix_fmts[i];
            else 
                ctx_p->pix_fmt = ctx_p->codec->pix_fmts[0];
        }

        /*如果未指定像素格式，则使用默认格式。*/
        if (ctx_p->pix_fmt == AV_PIX_FMT_NONE)
            ctx_p->pix_fmt = AV_PIX_FMT_YUV420P;

        /*No b frame.*/
        ctx_p->max_b_frames = 0;
        
    }
    else if(ctx_p->codec_type == AVMEDIA_TYPE_AUDIO)
    {
        assert(opt->time_base.den > 0 && opt->time_base.num > 0);
        assert(opt->sample_rate > 0);
        assert(opt->channels > 0);
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

     //   if (ctx_p->channel_layout == -1L)
     //       ctx_p->channel_layout = av_get_default_channel_layout(opt->channels);

        if (ctx_p->sample_fmt == AV_SAMPLE_FMT_NONE)
            ctx_p->sample_fmt = AV_SAMPLE_FMT_FLTP;

        if (ctx_p->codec_id == AV_CODEC_ID_AAC)
            av_dict_set(&dict_p, "strict", "-2", 0);
    }
    else 
    {
        goto final_error;//fix me.
    }

    /*如果流需要设置全局头部，则编码器需要知道这个请求。*/
    if (ctx->avctx->oformat->flags & AVFMT_GLOBALHEADER)
        ctx_p->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    
    chk = abcdk_avcodec_open(ctx_p, &dict_p);
    if(chk <0 )
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

int abcdk_ffmpeg_add_stream(abcdk_ffmpeg_t *ctx, const AVCodecContext *opt, int have_codec)
{
    AVStream *vs = NULL;
    int chk = -1;

    assert(ctx != NULL && opt != NULL);

    if (ctx->avctx->nb_streams >= ABCDK_FFMPEG_MAX_STREAMS)
        return -2;

    vs = abcdk_avformat_output_stream(ctx->avctx,abcdk_avcodec_find2(opt->codec_id,1));
    if(!vs)
        return -1;

    if (have_codec)
    {
        abcdk_avstream_parameters_from_context(vs, opt);

        /*如果流需要设置全局头部，则编码器需要知道这个请求。*/
        if (ctx->avctx->oformat->flags & AVFMT_GLOBALHEADER)
            vs->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    else
    {
        if (ctx->try_nvcodec)
        {
            /*NVCODEC硬件编码，必须用下面的写法，因为编码器可能未安装。*/
            if (opt->codec_id == AV_CODEC_ID_HEVC)
                chk = _abcdk_ffmpeg_writer_open_codec(ctx, vs->index, abcdk_avcodec_find("hevc_nvenc", 1), opt);
            else if (opt->codec_id == AV_CODEC_ID_H264)
                chk = _abcdk_ffmpeg_writer_open_codec(ctx, vs->index, abcdk_avcodec_find("h264_nvenc", 1), opt);
            else
                chk = -1;
        }

        /*如果硬件解码已经安装，到这里可能已经成功了。*/
        if (chk < 0)
            chk = _abcdk_ffmpeg_writer_open_codec(ctx, vs->index, abcdk_avcodec_find2(opt->codec_id, 1), opt);

        if (chk < 0)
            return -1;
    }

    return vs->index;
}

int abcdk_ffmpeg_write_header0(abcdk_ffmpeg_t *ctx,const AVDictionary *dict)
{
    int chk;

    assert(ctx != NULL);

    /*头部，写入一次就好。*/
    if(ctx->write_header_ok)
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

int abcdk_ffmpeg_write_header(abcdk_ffmpeg_t *ctx, int fmp4)
{
    AVDictionary *dict = NULL;
    int chk;

    av_dict_set(&dict, "movflags", "empty_moov+default_base_moof+frag_keyframe", 0);

    chk = abcdk_ffmpeg_write_header0(ctx,dict);

    av_dict_free(&dict);

    return chk;
}

int abcdk_ffmpeg_write_trailer(abcdk_ffmpeg_t *ctx)
{
    AVCodecContext *ctx_p = NULL;
    AVPacket pkt = {0};
    int chk;

    assert(ctx != NULL);

    /*写入头部后，才能写末尾。*/
    if(!ctx->write_header_ok)
        return -2;

    av_init_packet(&pkt);

    /* 写入所有延时编码数据包。*/
    for (int i = 0; i < ctx->avctx->nb_streams; i++)
    {
        ctx_p = ctx->codec_ctx[i];
        
        /*跳过使用外部编码器的流。*/
        if(!ctx_p)
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

            pkt.stream_index = ctx->avctx->streams[i]->index;

            chk = abcdk_ffmpeg_write(ctx, &pkt, NULL);
            if (chk < 0)
                goto final;
        }
    }

final:

    /*不要忘记。*/
    av_packet_unref(&pkt);

    chk = abcdk_avformat_output_trailer(ctx->avctx);
    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_ffmpeg_write(abcdk_ffmpeg_t *ctx, AVPacket *pkt, AVRational *src_time_base)
{
    AVRational bq,cq;
    AVCodecContext *ctx_p = NULL;
    AVStream *vs_p = NULL;
    int chk;

    assert(ctx != NULL && pkt != NULL);
    assert(pkt->stream_index >= 0 && pkt->stream_index < ctx->avctx->nb_streams);

    /*写入头部后，才能写正文。*/
    if(!ctx->write_header_ok)
        return -2;

    ctx_p = ctx->codec_ctx[pkt->stream_index];
    vs_p = ctx->avctx->streams[pkt->stream_index];

    /*如果没有源时间基值，则使用内部时间基值。*/
    if(src_time_base)
        bq = *src_time_base;
    else 
        bq = (ctx_p ? ctx_p->time_base : vs_p->codec->time_base);

    cq = vs_p->time_base;

    pkt->dts = av_rescale_q_rnd(pkt->dts, bq, cq, (enum AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
    pkt->pts = av_rescale_q_rnd(pkt->pts, bq, cq, (enum AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
	pkt->duration = av_rescale_q(pkt->duration, bq, cq);
    pkt->pos = -1;

    chk = abcdk_avformat_output_write(ctx->avctx, pkt);

    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_ffmpeg_write2(abcdk_ffmpeg_t *ctx, void *data, int size, int keyframe, int stream)
{
    AVPacket pkt = {0};
    AVStream *vs_p = NULL;
    int chk;

    assert(ctx != NULL && data != NULL && size > 0 && stream >= 0);
    assert(stream < ctx->avctx->nb_streams);

    vs_p = ctx->avctx->streams[stream];

    av_init_packet(&pkt);

    pkt.data = (uint8_t *)data;
    pkt.size = size;
    pkt.stream_index = stream;
    pkt.flags = (keyframe?AV_PKT_FLAG_KEY:0);
    pkt.dts = ++ctx->ts_nums[pkt.stream_index][1];
    pkt.pts = ++ctx->ts_nums[pkt.stream_index][0];

    chk = abcdk_ffmpeg_write(ctx, &pkt, NULL);
    if (chk < 0)
        return -1;

    return 0;
}


int abcdk_ffmpeg_write3(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream)
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

    frame_cp = av_frame_alloc();
    if(!frame_cp)
        return -1;

    frame_cp->width = frame->width;
    frame_cp->height = frame->height;
    frame_cp->format = frame->format;

    for (int i = 0; i < AV_NUM_DATA_POINTERS; i++)
    {
        if (frame->data[i] == NULL || frame->linesize[i] <= 0)
            break;

        frame_cp->data[i] = frame->data[i];
        frame_cp->linesize[i] = frame->linesize[i];
    }

    frame_cp->pts = ++ctx->ts_nums[stream][0];
    frame_cp->quality = frame->quality;

    av_init_packet(&pkt);
    chk = abcdk_avcodec_encode(ctx_p, &pkt, frame_cp);
    if (chk <= 0)
        goto final;

    pkt.stream_index = stream;
    chk = abcdk_ffmpeg_write(ctx, &pkt,0);

final:

    av_packet_unref(&pkt);
    av_frame_free(&frame_cp);
    

    return chk;
}

#pragma GCC diagnostic pop

#endif //AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

