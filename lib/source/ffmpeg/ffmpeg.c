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

    /** 视频。*/
    AVFormatContext *avctx;

    /** 视频字典。*/
    AVDictionary *dict;

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

AVFormatContext *abcdk_ffmpeg_ctxptr(abcdk_ffmpeg_t *ctx)
{
    assert(ctx != NULL);

    return ctx->avctx;
}

int64_t _abcdk_ffmpeg_clock()
{
    abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0);
}

int _abcdk_ffmpeg_capture_interrupt_cb(void *args)
{
    abcdk_ffmpeg_t *ctx = (abcdk_ffmpeg_t *)args;
    uint64_t cur_time = _abcdk_ffmpeg_clock();

    if (ctx->timeout > 0)
    {
        /* 如果超时，返回失败。*/
        if ((cur_time - ctx->last_packet_time) >= ctx->timeout)
            return -1;
    }

    return 0;
}

abcdk_ffmpeg_t *abcdk_ffmpeg_open_capture(const char *short_name, const char *url,AVIOContext *io,time_t timeout)
{
    abcdk_ffmpeg_t *ctx = NULL;
    int is_mp4_file = 0;
    int chk;

    assert(url != NULL);

    ctx= abcdk_heap_alloc(sizeof(abcdk_ffmpeg_t));
    if(!ctx)
        return NULL;

    ctx->timeout = timeout;
    ctx->last_packet_time = _abcdk_ffmpeg_clock();

    AVIOInterruptCB cb;
    cb.callback = _abcdk_ffmpeg_capture_interrupt_cb;
    cb.opaque = ctx;

    av_init_packet(&ctx->read_pkt);

    ctx->avctx = abcdk_avformat_input_open(short_name,url,&cb,io,&ctx->dict);
    if(!ctx->avctx)
        goto final_error;

    chk = abcdk_avformat_input_probe(ctx->avctx, NULL);
    if (chk < 0)
        goto final_error;

    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "QuickTime / MOV") == 0 ? 0x1 : 0);
    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "FLV (Flash Video)") == 0 ? 0x2 : 0);
    is_mp4_file |= (strcmp(ctx->avctx->iformat->long_name, "Matroska / WebM") == 0 ? 0x4 : 0);

    for(int i = 0;i<ctx->avctx->nb_streams;i++)
    {
        ctx->input_mp4_h264[i] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_H264 && is_mp4_file);
        ctx->input_mp4_h265[i] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_HEVC && is_mp4_file);
        ctx->input_mp4_mpeg4[i] = (ctx->avctx->streams[i]->codec->codec_id == AV_CODEC_ID_MPEG4 && is_mp4_file);
    }

    return ctx;

final_error:

    abcdk_ffmpeg_destroy(&ctx);

    return NULL;
}

int _abcdk_ffmpeg_open_capture_codec(abcdk_ffmpeg_t *ctx, int stream, AVCodec *codec)
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

    /*优先尝试硬件解码。必须用下面的写法，因为解码器可能未安装。*/
    if (codecpar->codec_id == AV_CODEC_ID_HEVC)
        _abcdk_ffmpeg_open_capture_codec(ctx, stream, abcdk_avcodec_find("hevc_cuvid", 0));
    else if (codecpar->codec_id == AV_CODEC_ID_H264)
        _abcdk_ffmpeg_open_capture_codec(ctx, stream, abcdk_avcodec_find("h264_cuvid", 0));

    /*如果硬件解码已经安装，到这里可能已经成功了。*/
    codec_ctx_p = ctx->codec_ctx[stream];
    if (codec_ctx_p)
        return 0;

    chk = _abcdk_ffmpeg_open_capture_codec(ctx, stream, abcdk_avcodec_find2(codecpar->codec_id, 0));
    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_ffmpeg_read(abcdk_ffmpeg_t *ctx, AVPacket *packet, int stream)
{
    uint8_t *extdata_p = NULL;
    int extsize = 0;
    int oldsize = 0;
    int chk;

    assert(ctx != NULL && packet != NULL);

    for (;;)
    {
        chk = abcdk_avformat_input_read(ctx->avctx, packet, AVMEDIA_TYPE_NB);
        if (chk < 0)
            return -1;
        
        /*读数据包 +1。*/
        ctx->read_pkt_count[packet->stream_index] += 1;

        /* 更新最近包时间，不然会超时。*/
        ctx->last_packet_time = _abcdk_ffmpeg_clock();

        /* 如果指定了流索引，这里筛一下。*/
        if (stream >= 0)
        {
            if(packet->stream_index != stream)
                continue;
        }
        
        if (ctx->input_mp4_mpeg4[packet->stream_index]) 
        {
            /*mp4格式的mpeg码流需要特殊处理一下。*/

            // if (ctx->read_pkt_count[packet->stream_index] == 1)
            // {
            //     extdata_p = ctx->avctx->streams[packet->stream_index]->codec->extradata;
            //     extsize = ctx->avctx->streams[packet->stream_index]->codec->extradata_size;

            //     if (extsize > 0)
            //     {
            //         /*记录现有数据长度。*/
            //         oldsize = packet->size;
            //         /*mpeg全局头部有三个字节(0x00 0x00 0x01)的启起码，因此要减去。*/
            //         av_grow_packet(packet, extsize - 3);
            //         /*把现有数据向后移动。*/
            //         memmove(packet->data + (extsize - 3), packet->data, oldsize);
            //         /*复制全局数据到开头。*/
            //         memcpy(packet->data, extdata_p + 3,extsize - 3);
            //     }
            // }
        }
        else if (ctx->input_mp4_h264[packet->stream_index] || ctx->input_mp4_h265[packet->stream_index])
        {
            /*只有mp4格式的h264、h265才需要执行下面的过滤器。*/

            chk = abcdk_avformat_input_filter(ctx->avctx, packet, &ctx->vs_filter[packet->stream_index]);
            if (chk < 0)
                return -1;
        }

        break;
    }

    return packet->stream_index;
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

abcdk_ffmpeg_t *abcdk_ffmpeg_open_writer(const char*short_name,const char *url,const char *mime_type,AVIOContext *io)
{
    abcdk_ffmpeg_t *ctx = NULL;

    assert(url!= NULL);

    ctx = abcdk_heap_alloc(sizeof(abcdk_ffmpeg_t));
    if(!ctx)
        return NULL;

    ctx->avctx = abcdk_avformat_output_open(short_name, url, mime_type, NULL, io);
    if(!ctx->avctx)
        goto final_error;

    return ctx;

final_error:

    abcdk_ffmpeg_destroy(&ctx);

    return NULL;
}

int _abcdk_ffmpeg_open_writer_codec(abcdk_ffmpeg_t *ctx, int stream, int fps, int width, int height,AVCodec *codec)
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
        abcdk_avcodec_video_encode_prepare(ctx_p, fps, width, height, -1, ctx->avctx->oformat->flags);

   //     ctx_p->thread_count = 2;
        ctx_p->max_b_frames = 0;
    }
    else 
    {
        goto final_error;//fix me.
    }
    
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

int abcdk_ffmpeg_add_stream(abcdk_ffmpeg_t *ctx, int fps, int width, int height, enum AVCodecID id,
                            const void *extdata, int extsize, int have_codec)
{
    AVStream *vs = NULL;
    int chk;

    assert(ctx != NULL && fps > 0 && width > 0 && height > 0 && id > AV_CODEC_ID_NONE);

    if (ctx->avctx->nb_streams >= ABCDK_FFMPEG_MAX_STREAMS)
        return -2;

    vs = abcdk_avformat_output_stream(ctx->avctx,abcdk_avcodec_find2(id,1));
    if(!vs)
        return -1;

    if (have_codec)
    {
        /* 使用外部编码器时，下面值必须填写。*/

        vs->time_base = vs->codec->time_base = av_make_q(1, fps);
        vs->avg_frame_rate = vs->r_frame_rate = av_make_q(fps, 1);

        if (ctx->avctx->oformat->flags & AVFMT_GLOBALHEADER)
            vs->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        vs->codec->width = width;
        vs->codec->height = height;

        /*如果有扩展信息，必须复制，不然流无法解码。*/
        if (extdata != NULL && extsize > 0)
        {
            if(vs->codec->extradata)
                av_free(vs->codec->extradata);
            
            vs->codec->extradata = NULL;
            vs->codec->extradata_size = extsize;
            vs->codec->extradata = (uint8_t *)av_mallocz((size_t)(extsize + AV_INPUT_BUFFER_PADDING_SIZE));
            memcpy(vs->codec->extradata, extdata, extsize);
        }

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58,35,100) 
        vs->codecpar->width = width;
        vs->codecpar->height = height;

        /*如果有扩展信息，必须复制，不然流无法解码。*/
        if (extdata != NULL && extsize > 0)
        {
            if(vs->codecpar->extradata)
                av_free(vs->codecpar->extradata);
            
            vs->codecpar->extradata = NULL;
            vs->codecpar->extradata_size = extsize;
            vs->codecpar->extradata = (uint8_t *)av_mallocz((size_t)(extsize + AV_INPUT_BUFFER_PADDING_SIZE));
            memcpy(vs->codecpar->extradata, extdata, extsize);
        }
#endif
    }
    else
    {
        /*优先尝试硬件编码。必须用下面的写法，因为编码器可能未安装。*/
        if (id == AV_CODEC_ID_HEVC)
            chk = _abcdk_ffmpeg_open_writer_codec(ctx,vs->index,fps,width,height,abcdk_avcodec_find("hevc_nvenc",1));
        else if (id == AV_CODEC_ID_H264)
            chk = _abcdk_ffmpeg_open_writer_codec(ctx,vs->index,fps,width,height,abcdk_avcodec_find("h264_nvenc",1));
        else 
            chk = -1;

        if(chk<0)
            chk = _abcdk_ffmpeg_open_writer_codec(ctx,vs->index,fps,width,height,abcdk_avcodec_find2(id,1));

        if (chk < 0)
            return -1;
    }

    return 0;
}

int abcdk_ffmpeg_write_header(abcdk_ffmpeg_t *ctx, int fmp4)
{
    int chk;

    assert(ctx != NULL);

    /*头部，写入一次就好。*/
    if(ctx->write_header_ok)
        return 0;

    if (fmp4)
        av_dict_set(&ctx->dict, "movflags", "empty_moov+default_base_moof+frag_keyframe", 0);

    chk = abcdk_avformat_output_header(ctx->avctx, &ctx->dict);
    if (chk < 0)
        return -1;

    /*Set OK.*/
    ctx->write_header_ok = 1;

    return 0;
}

int abcdk_ffmpeg_write_trailer(abcdk_ffmpeg_t *ctx)
{
    AVCodecContext *ctx_p = NULL;
    AVPacket pkt;
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

            chk = abcdk_ffmpeg_write(ctx, &pkt);
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

int abcdk_ffmpeg_write(abcdk_ffmpeg_t *ctx, AVPacket *packet)
{
    AVRational bq,cq;
    AVCodecContext *ctx_p = NULL;
    AVStream *vs_p = NULL;
    int chk;

    assert(ctx != NULL && packet != NULL);
    assert(packet->stream_index >= 0 && packet->stream_index < ctx->avctx->nb_streams);

    /*写入头部后，才能写正文。*/
    if(!ctx->write_header_ok)
        return -2;

    ctx_p = ctx->codec_ctx[packet->stream_index];
    vs_p = ctx->avctx->streams[packet->stream_index];

    bq = (ctx_p ? ctx_p->time_base : vs_p->codec->time_base);
    cq = vs_p->time_base;
    packet->stream_index = vs_p->index;

    chk = abcdk_avformat_output_write(ctx->avctx, &bq, &cq, packet);
    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_ffmpeg_write2(abcdk_ffmpeg_t *ctx, void *data, int size, int keyframe, int stream)
{
    AVPacket pkt;
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

    /*No B-frame.*/
    pkt.dts = ++ctx->ts_nums[stream][1];
    pkt.pts = ++ctx->ts_nums[stream][0];

    chk = abcdk_ffmpeg_write(ctx, &pkt);
    if (chk < 0)
        return -1;

    return 0;
}


int abcdk_ffmpeg_write3(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream)
{
    AVCodecContext *ctx_p = NULL;
    AVStream *vs_p = NULL;
    AVFrame *frame_cp = NULL;
    AVPacket pkt;
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
    chk = abcdk_ffmpeg_write(ctx, &pkt);

final:

    av_packet_unref(&pkt);
    av_frame_free(&frame_cp);
    

    return chk;
}

#pragma GCC diagnostic pop

#endif //AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

