/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/video.h"

#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H) && defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

/*------------------------------------------------------------------------------------------------*/


void abcdk_video_close(abcdk_video_t *video)
{
    if(!video)
        return;

    for (int i = 0; i < ABCDK_VIDEO_MAX_STREAMS; i++)
    {
        if(video->codec_ctx[i])
            abcdk_avcodec_free(&video->codec_ctx[i]);

        if(video->codec_dict[i])
            av_dict_free(&video->codec_dict[i]);

        if(video->vs_filter[i])
        {
            av_bitstream_filter_close(video->vs_filter[i]);
            video->vs_filter[i] = NULL;
        }
    }

    av_dict_free(&video->dict);
    abcdk_avformat_free(&video->ctx);

    abcdk_heap_free(video);
}

int abcdk_video_nb_streams(abcdk_video_t *video)
{
    assert(video != NULL);

    return video->ctx->nb_streams;
}

int abcdk_video_check_stream(abcdk_video_t *video,int stream_index,int type)
{
    AVStream *vs_p = NULL;
    int chk = -1;

    assert(video != NULL && stream_index >= 0 && type >= 1 && type <= 3);
    assert(video->ctx->nb_streams > stream_index);

    vs_p = video->ctx->streams[stream_index];

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

int abcdk_video_find_stream(abcdk_video_t *video,int type)
{
    int nb_streams = 0;
    int chk;

    assert(video != NULL && type >= 1 && type <= 3);

    nb_streams = abcdk_video_nb_streams(video);

    for (int i = 0; i < nb_streams; i++)
    {
        chk = abcdk_video_check_stream(video,i,type);
        if(chk==0)
            return i;
    }

    return -1;
}

double abcdk_video_get_duration(abcdk_video_t *video, int stream_index)
{
    AVStream *vs_p = NULL;

    assert(video != NULL && stream_index >= 0);
    assert(video->ctx->nb_streams > stream_index);

    vs_p = video->ctx->streams[stream_index];

    return abcdk_avstream_get_duration(video->ctx,vs_p);
}

int abcdk_video_get_width(abcdk_video_t *video, int stream_index)
{
    AVStream *vs_p = NULL;

    assert(video != NULL && stream_index >= 0);
    assert(video->ctx->nb_streams > stream_index);

    vs_p = video->ctx->streams[stream_index];

    return vs_p->codec->width;
}

int abcdk_video_get_height(abcdk_video_t *video, int stream_index)
{
    AVStream *vs_p = NULL;

    assert(video != NULL && stream_index >= 0);
    assert(video->ctx->nb_streams > stream_index);

    vs_p = video->ctx->streams[stream_index];

    return vs_p->codec->height;
}

double abcdk_video_get_fps(abcdk_video_t *video, int stream_index)
{
    AVStream *vs_p = NULL;

    assert(video != NULL && stream_index >= 0);
    assert(video->ctx->nb_streams > stream_index);

    vs_p = video->ctx->streams[stream_index];

    return abcdk_avstream_get_fps(video->ctx,vs_p);
}

double abcdk_video_ts2sec(abcdk_video_t *video, int stream_index, int64_t ts)
{
    AVStream *vs_p = NULL;

    assert(video != NULL && stream_index >= 0);
    assert(video->ctx->nb_streams > stream_index);

    vs_p = video->ctx->streams[stream_index];

    return abcdk_avstream_ts2sec(video->ctx,vs_p,ts);
}

int _abcdk_video_capture_interrupt_cb(void *args)
{
    abcdk_video_t *video = (abcdk_video_t *)args;
    uint64_t cur_time = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0);

    if (video->timeout > 0)
    {
        /* 如果超时，返回失败。*/
        if ((cur_time - video->last_packet_time) >= video->timeout)
            return -1;
    }

    return 0;
}

abcdk_video_t *abcdk_video_open_capture(const char *short_name, const char *url, int64_t timeout, int dump)
{
    abcdk_video_t *video = NULL;
    int chk;

    assert(url != NULL);

    video= abcdk_heap_alloc(sizeof(abcdk_video_t));
    if(!video)
        return NULL;

    video->timeout = timeout;
    video->last_packet_time = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,0);

    AVIOInterruptCB cb;
    cb.callback = _abcdk_video_capture_interrupt_cb;
    cb.opaque = video;

    video->ctx = abcdk_avformat_input_open(short_name,url,&cb,NULL,&video->dict);
    if(!video->ctx)
        goto final_error;

    chk = abcdk_avformat_input_probe(video->ctx, NULL, dump);
    if (chk < 0)
        goto final_error;

    return video;

final_error:

    abcdk_heap_free(video);

    return NULL;
}

int _abcdk_video_open_capture_codec(abcdk_video_t *video, int stream_index,const char *codec_name)
{
    AVCodecContext *ctx_p = NULL;
    AVDictionary *dict_p = NULL;
    AVStream *vs_p = NULL;
    int chk;

    if (stream_index >= video->ctx->nb_streams)
        return -1;

    if (stream_index >= ABCDK_VIDEO_MAX_STREAMS)
        return -1;

    /* 如果已经打开，直接返回。*/
    if(video->codec_ctx[stream_index])
        return 0;

    vs_p = video->ctx->streams[stream_index];

    if(codec_name)
        ctx_p = abcdk_avcodec_alloc2(codec_name,0);
    else 
        ctx_p = abcdk_avcodec_alloc3(vs_p->codec->codec_id,0);

    if(!ctx_p)
        goto final_error;
    
    abcdk_avstream_parameters_to_context(ctx_p, vs_p);
    
    chk = abcdk_avcodec_open(ctx_p, &dict_p);
    if(chk <0 )
        goto final_error;

    video->codec_ctx[stream_index] = ctx_p;
    video->codec_dict[stream_index] = dict_p;

    return 0;

final_error:

    abcdk_avcodec_free(&ctx_p);
    av_dict_free(&dict_p);

    return -1;
}

int abcdk_video_read(abcdk_video_t *video, AVPacket *pkt, int stream_index, int only_key, int not_filter)
{
    int chk;
    assert(video != NULL && pkt != NULL);

    for (;;)
    {
        chk = abcdk_avformat_input_read(video->ctx, pkt, AVMEDIA_TYPE_NB);
        if (chk < 0)
            return -1;

        /* 更新最近包时间，不然会超时。*/
        video->last_packet_time = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0);

        if (stream_index >= 0)
        {
            if(pkt->stream_index != stream_index)
                continue;
        }

        if(!not_filter)
        {
            chk = abcdk_avformat_input_filter(video->ctx,pkt,&video->vs_filter[pkt->stream_index]);
            if (chk < 0)
                return -1;
        }

        if (!only_key || (pkt->flags & AV_PKT_FLAG_KEY))
            break;
     }

    return pkt->stream_index;
}

int abcdk_video_read2(abcdk_video_t *video, AVFrame *fae, int stream_index, int only_key)
{
    AVStream *vs_p = NULL;
    AVCodecContext *codec_ctx_p;
    AVPacket pkt;
    int chk;

    assert(video != NULL && fae != NULL);

    av_frame_unref(fae);
    av_init_packet(&pkt);

    for (;;)
    {
        chk = abcdk_video_read(video, &pkt, stream_index, only_key, 0);
        if (chk < 0)
            return -1;

        vs_p = video->ctx->streams[pkt.stream_index];

        /*优先尝试硬件解码。*/
        if (vs_p->codec->codec_id == AV_CODEC_ID_HEVC)
            chk = _abcdk_video_open_capture_codec(video, pkt.stream_index, "hevc_cuvid");
        else if (vs_p->codec->codec_id == AV_CODEC_ID_H264)
            chk = _abcdk_video_open_capture_codec(video, pkt.stream_index, "h264_cuvid");

        if (chk < 0)
            chk = _abcdk_video_open_capture_codec(video, pkt.stream_index, NULL);

        if (chk < 0)
            goto final;

        codec_ctx_p = video->codec_ctx[pkt.stream_index];
        
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

abcdk_video_t *abcdk_video_open_writer(const char*short_name,const char *url,const char *mime_type)
{
    abcdk_video_t *video = NULL;

    assert(url!= NULL);

    video = abcdk_heap_alloc(sizeof(abcdk_video_t));
    if(!video)
        return NULL;

    video->ctx = abcdk_avformat_output_open(short_name, url, mime_type, NULL, NULL, NULL);
    if(!video->ctx)
        goto final_error;

    return video;

final_error:

    abcdk_heap_free(video);

    return NULL;
}

int _abcdk_video_open_writer_codec(abcdk_video_t *video, int stream_index,
                                   int fps, int width, int height,const char *codec_name)
{
    AVCodecContext *ctx_p = NULL;
    AVDictionary *dict_p = NULL;
    AVStream *vs_p = NULL;
    int chk;

    if (stream_index >= video->ctx->nb_streams)
        return -1;

    if (stream_index >= ABCDK_VIDEO_MAX_STREAMS)
        return -1;

    /* 如果已经打开，直接返回。*/
    if(video->codec_ctx[stream_index])
        return 0;

    vs_p = video->ctx->streams[stream_index];

    if (codec_name)
        ctx_p = abcdk_avcodec_alloc2(codec_name, 1);
    else
        ctx_p = abcdk_avcodec_alloc3(vs_p->codec->codec_id, 1);

    if(!ctx_p)
        goto final_error;

    if(ctx_p->codec_type == AVMEDIA_TYPE_VIDEO)
    {
        abcdk_avcodec_video_encode_prepare(ctx_p, fps, width, height, -1, video->ctx->oformat->flags);

        ctx_p->thread_count = 2;
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

    video->codec_ctx[stream_index] = ctx_p;
    video->codec_dict[stream_index] = dict_p;

    return 0;

final_error:

    abcdk_avcodec_free(&ctx_p);
    av_dict_free(&dict_p);

    return -1;
}

int abcdk_video_add_stream(abcdk_video_t *video, int fps, int width, int height, enum AVCodecID id,
                           const void *extdata, int extsize, int have_codec)
{
    AVCodecContext *ctx_p = NULL;
    AVDictionary *dict_p = NULL;
    AVStream *vs = NULL;
    int chk;

    assert(video != NULL && fps > 0 && width > 0 && height > 0 && id > AV_CODEC_ID_NONE);

    if (video->ctx->nb_streams >= ABCDK_VIDEO_MAX_STREAMS)
        return -2;

    vs = abcdk_avformat_output_stream3(video->ctx,id);
    if(!vs)
        return -1;

    if (have_codec)
    {
        if (video->ctx->oformat->flags & AVFMT_GLOBALHEADER)
            vs->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        vs->time_base = vs->codec->time_base = av_make_q(1, fps);
        vs->avg_frame_rate = vs->r_frame_rate = av_make_q(fps, 1);
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
    }
    else
    {
        /*优先尝试硬件编码。*/
        if (id == AV_CODEC_ID_HEVC)
            chk = _abcdk_video_open_writer_codec(video,vs->index,fps,width,height,"hevc_nvenc");
        else if (id == AV_CODEC_ID_H264)
            chk = _abcdk_video_open_writer_codec(video,vs->index,fps,width,height,"h264_nvenc");
        
        if(chk<0)
            chk = _abcdk_video_open_writer_codec(video,vs->index,fps,width,height,NULL);

        if (chk < 0)
            return -1;
    }

    return 0;
}

int abcdk_video_write_header(abcdk_video_t *video, int make_mp4fragment, int dump)
{
    int chk;

    assert(video != NULL);

    if (make_mp4fragment)
        av_dict_set(&video->dict, "movflags", "empty_moov+default_base_moof+frag_keyframe", 0);

    chk = abcdk_avformat_output_header(video->ctx, &video->dict, dump);
    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_video_write_trailer(abcdk_video_t *video)
{
    int chk;

    assert(video != NULL);

    /* 写入所有延时编码数据包。*/
    for (int i = 0; i < video->ctx->nb_streams; i++)
        abcdk_video_write2(video, i, NULL);

    chk = abcdk_avformat_output_trailer(video->ctx);
    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_video_write(abcdk_video_t *video, AVPacket *pkt)
{
    AVStream *vs_p = NULL;
    int chk;

    assert(video != NULL && pkt != NULL);
    assert(pkt->stream_index >= 0 && pkt->stream_index < video->ctx->nb_streams);

    vs_p = video->ctx->streams[pkt->stream_index];

    chk = abcdk_avformat_output_write(video->ctx, vs_p, pkt);
    if (chk < 0)
        return -1;

    return 0;
}

int abcdk_video_write2(abcdk_video_t *video,int stream_index, AVFrame *fae)
{
    AVCodecContext *ctx_p = NULL;
    AVStream *vs_p = NULL;
    AVPacket pkt;
    AVFrame *fae_cp = NULL;
    int chk;

    assert(video != NULL && stream_index >= 0);
    assert(stream_index < video->ctx->nb_streams);

    ctx_p = video->codec_ctx[stream_index];
    vs_p = video->ctx->streams[stream_index];

    /*使用外部编器，不支持。*/
    if (!ctx_p)
        return -2;

    av_init_packet(&pkt);

    if (fae != NULL)
    {
        fae_cp = av_frame_alloc();
        fae_cp->width = fae->width;
        fae_cp->height = fae->height;
        fae_cp->format = fae->format;
        fae_cp->data[0] = fae->data[0];
        fae_cp->data[1] = fae->data[1];
        fae_cp->data[2] = fae->data[2];
        fae_cp->data[3] = fae->data[3];
        fae_cp->linesize[0] = fae->linesize[0];
        fae_cp->linesize[1] = fae->linesize[1];
        fae_cp->linesize[2] = fae->linesize[2];
        fae_cp->linesize[3] = fae->linesize[3];

        fae_cp->pts = ++video->ts_nums[stream_index][0];
        fae_cp->quality = 1;
    }

    do
    {
        if (!fae)
        {
            /* 检查当前编码器是否支持延时编码。*/
            if (!(vs_p->codec->codec->capabilities & AV_CODEC_CAP_DELAY))
                break;
        }

        chk = abcdk_avcodec_encode(ctx_p, &pkt, (fae ? fae_cp : NULL));
        if (chk <= 0)
            break;

        pkt.stream_index = stream_index;

        chk = abcdk_video_write(video, &pkt);
        if(chk < 0)
            break;

    } while (!fae);

    av_frame_free(&fae_cp);
    av_packet_unref(&pkt);

    return chk;
}

int abcdk_video_write3(abcdk_video_t *video, int stream_index, void *data, int size)
{
    AVPacket pkt;
    AVStream *vs_p = NULL;
    int chk;

    assert(video != NULL && stream_index >= 0 && data != NULL && size > 0);
    assert(stream_index < video->ctx->nb_streams);

    vs_p = video->ctx->streams[stream_index];

    av_init_packet(&pkt);

    pkt.data = (uint8_t *)data;
    pkt.size = size;
    pkt.stream_index = stream_index;

    /*No B-frame.*/
    pkt.dts = ++video->ts_nums[stream_index][1];
    pkt.pts = ++video->ts_nums[stream_index][0];

    chk = abcdk_video_write(video, &pkt);
    if (chk < 0)
        return -1;

    return 0;
}

#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H && AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

