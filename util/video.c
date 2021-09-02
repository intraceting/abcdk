/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/video.h"

#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H) && defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

/** 最大支持16个。*/
#define ABCDK_VIDEO_MAX_STREAMS     16

/**
 * 视频捕获环境。
*/
typedef struct _abcdk_video_capture
{
    /** 解码器环境。*/
    AVCodecContext *codec_ctx[ABCDK_VIDEO_MAX_STREAMS];

    /** 解码器字典。*/
    AVDictionary *codec_dict[ABCDK_VIDEO_MAX_STREAMS];

    /** 数据包过滤器。*/
    AVBitStreamFilterContext *vs_filter[ABCDK_VIDEO_MAX_STREAMS];

    /** 流环境。*/
    AVFormatContext *ctx;

    /** 流字典。*/
    AVDictionary *dict;

    /** 超时(秒)。*/
    uint64_t timeout;

    /** 最近捕获包时间(秒)。*/
    uint64_t last_packet_time;

} abcdk_video_capture_t;

int _abcdk_video_capture_interrupt_cb(void *args)
{
    abcdk_video_capture_t *vc = (abcdk_video_capture_t *)args;
    uint64_t cur_time = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0);

    if (vc->timeout > 0)
    {
        /* 如果超时，返回失败。*/
        if ((cur_time - vc->last_packet_time) >= vc->timeout)
            return -1;
    }

    return 0;
}

void abcdk_video_capture_close(abcdk_video_capture_t *vc)
{
    if(!vc)
        return;

    for (int i = 0; i < ABCDK_VIDEO_MAX_STREAMS; i++)
    {
        if(vc->codec_ctx[i])
            abcdk_avcodec_free(&vc->codec_ctx[i]);

        if(vc->codec_dict[i])
            av_dict_free(&vc->codec_dict[i]);

        if(vc->vs_filter[i])
        {
            av_bitstream_filter_close(vc->vs_filter[i]);
            vc->vs_filter[i] = NULL;
        }
    }

    av_dict_free(&vc->dict);
    abcdk_avformat_free(&vc->ctx);

    abcdk_heap_free(vc);
}

int abcdk_video_capture_nb_streams(abcdk_video_capture_t *vc)
{
    assert(vc != NULL);

    return vc->ctx->nb_streams;
}

int abcdk_video_capture_check_stream(abcdk_video_capture_t *vc,int stream_index,int type)
{
    AVStream *vs_p = NULL;
    int chk = -1;

    assert(vc != NULL && stream_index >= 0 && type >= 1 && type <= 3);
    assert(vc->ctx->nb_streams > stream_index);

    vs_p = vc->ctx->streams[stream_index];

    if (type == 1)
        chk = ((vs_p->codec->codec_type == AVMEDIA_TYPE_VIDEO) ? 0 : -1);
    else if (type == 2)
        chk = ((vs_p->codec->codec_type == AVMEDIA_TYPE_AUDIO) ? 0 : -1);
    else if (type == 3)
        chk = ((vs_p->codec->codec_type == AVMEDIA_TYPE_SUBTITLE) ? 0 : -1);
    else 
        chk = -2;
    
    return chk;
}

int abcdk_video_capture_find_stream(abcdk_video_capture_t *vc,int type)
{
    int nb_streams = 0;
    int chk;

    assert(vc != NULL && type >= 1 && type <= 3);

    nb_streams = abcdk_video_capture_nb_streams(vc);

    for (int i = 0; i < nb_streams; i++)
    {
        chk = abcdk_video_capture_check_stream(vc,i,type);
        if(chk==0)
            return i;
    }

    return -1;
}

double abcdk_video_capture_get_duration(abcdk_video_capture_t *vc, int stream_index)
{
    AVStream *vs_p = NULL;

    assert(vc != NULL && stream_index >= 0);
    assert(vc->ctx->nb_streams > stream_index);

    vs_p = vc->ctx->streams[stream_index];

    return abcdk_avstream_get_duration(vc->ctx,vs_p);
}

double abcdk_video_capture_get_fps(abcdk_video_capture_t *vc, int stream_index)
{
    AVStream *vs_p = NULL;

    assert(vc != NULL && stream_index >= 0);
    assert(vc->ctx->nb_streams > stream_index);

    vs_p = vc->ctx->streams[stream_index];

    return abcdk_avstream_get_fps(vc->ctx,vs_p);
}

double abcdk_video_capture_ts2sec(abcdk_video_capture_t *vc, int stream_index, int64_t ts)
{
    AVStream *vs_p = NULL;

    assert(vc != NULL && stream_index >= 0);
    assert(vc->ctx->nb_streams > stream_index);

    vs_p = vc->ctx->streams[stream_index];

    return abcdk_avstream_ts2sec(vc->ctx,vs_p,ts);
}

abcdk_video_capture_t *abcdk_video_capture_open(const char *short_name, const char *url, int64_t timeout, int dump)
{
    abcdk_video_capture_t *vc = NULL;
    int chk;

    assert(url != NULL);

    vc= abcdk_heap_alloc(sizeof(abcdk_video_capture_t));
    if(!vc)
        return NULL;

    vc->timeout = timeout;
    vc->last_packet_time = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,0);

    AVIOInterruptCB cb;
    cb.callback = _abcdk_video_capture_interrupt_cb;
    cb.opaque = vc;

    vc->ctx = abcdk_avformat_input_open(short_name,url,&cb,NULL,&vc->dict);
    if(!vc->ctx)
        goto final_error;

    chk = abcdk_avformat_input_probe(vc->ctx, NULL, dump);
    if (chk < 0)
        goto final_error;

    return vc;

final_error:

    abcdk_heap_free(vc);

    return NULL;
}

int _abcdk_video_capture_open_codec(abcdk_video_capture_t *vc, int stream_index)
{
    AVCodecContext *ctx_p = NULL;
    AVDictionary *dict_p = NULL;
    AVStream *vs_p = NULL;
    int chk;

    if (stream_index >= vc->ctx->nb_streams)
        return -1;

    if (stream_index >= ABCDK_VIDEO_MAX_STREAMS)
        return -1;

    /* 如果已经打开，直接返回。*/
    if(vc->codec_ctx[stream_index])
        return 0;

    vs_p = vc->ctx->streams[stream_index];

    ctx_p = abcdk_avcodec_alloc3(vs_p->codec->codec_id,0);
    if(!ctx_p)
        goto final_error;
    
    abcdk_avstream_parameters_to_context(ctx_p, vs_p);
    
    chk = abcdk_avcodec_open(ctx_p, &dict_p);
    if(chk <0 )
        goto final_error;

    vc->codec_ctx[stream_index] = ctx_p;
    vc->codec_dict[stream_index] = dict_p;

    return 0;

final_error:

    abcdk_avcodec_free(&ctx_p);
    av_dict_free(&dict_p);

    return -1;
}

int abcdk_video_capture_read(abcdk_video_capture_t *vc, AVPacket *pkt, int stream_index, int only_key)
{
    int chk;
    assert(vc != NULL && pkt != NULL);

    for (;;)
    {
        chk = abcdk_avformat_input_read(vc->ctx, pkt, AVMEDIA_TYPE_NB);
        if (chk < 0)
            return -1;

        /* 更新最近包时间，不然会超时。*/
        vc->last_packet_time = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0);

        if (stream_index >= 0)
        {
            if(pkt->stream_index != stream_index)
                continue;
        }

        chk = abcdk_avformat_input_filter(vc->ctx,pkt,&vc->vs_filter[pkt->stream_index]);
        if (chk < 0)
            return -1;

        if (!only_key || (pkt->flags & AV_PKT_FLAG_KEY))
            break;
     }

    return pkt->stream_index;
}

int abcdk_video_capture_read2(abcdk_video_capture_t *vc, AVFrame *fae, int stream_index, int only_key)
{
    AVCodecContext *codec_ctx_p;
    AVPacket pkt;
    int chk;

    assert(vc != NULL && fae != NULL);

    av_frame_unref(fae);
    av_init_packet(&pkt);

    for (;;)
    {
        chk = abcdk_video_capture_read(vc, &pkt, stream_index, only_key);
        if (chk < 0)
            return -1;

        chk = _abcdk_video_capture_open_codec(vc, pkt.stream_index);
        if (chk < 0)
            goto final;

        codec_ctx_p = vc->codec_ctx[pkt.stream_index];
        chk = abcdk_avcodec_decode(codec_ctx_p, fae, &pkt);

        if (chk != 0)
        {
            if (chk > 0)
                chk = pkt.stream_index;
            
            /*退出循环。*/
            break;
        }
    }
   

final:

    av_packet_unref(&pkt);

    return chk;
}

int abcdk_video_capture_read3(abcdk_video_capture_t *vc, abcdk_image_t *img, int stream_index, int only_key)
{
    
}


#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H && AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

