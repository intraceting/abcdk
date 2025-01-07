/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/ffmpeg/avcodec.h"

#ifdef AVCODEC_AVCODEC_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

AVCodec *abcdk_avcodec_find(const char *name,int encode)
{
    AVCodec *ctx = NULL;

    assert(name != NULL);
    assert(*name != '\0');

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    avcodec_register_all();
#endif

    ctx =(AVCodec *)(encode ? avcodec_find_encoder_by_name(name) : avcodec_find_decoder_by_name(name));

    return ctx;
}

AVCodec *abcdk_avcodec_find2(enum AVCodecID id,int encode)
{
    AVCodec *ctx = NULL;

    assert(id > AV_CODEC_ID_NONE);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    avcodec_register_all();
#endif

    ctx = (AVCodec *)(encode ? avcodec_find_encoder(id) : avcodec_find_decoder(id));
    
    return ctx;
}

void abcdk_avcodec_show_options(AVCodec *ctx)
{
    assert(ctx != NULL);

    if (ctx->priv_class)
        av_opt_show2((void *)&ctx->priv_class, NULL, -1, 0);
    else
        av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->long_name ? ctx->long_name : ctx->name));
}

void abcdk_avcodec_free(AVCodecContext **ctx)
{
    assert(ctx != NULL);

    if (*ctx)
        avcodec_close(*ctx);

    avcodec_free_context(ctx);
}

AVCodecContext *abcdk_avcodec_alloc(const AVCodec *ctx)
{
    assert(ctx != NULL);

    return avcodec_alloc_context3(ctx);
}

AVCodecContext *abcdk_avcodec_alloc2(const char *name,int encode)
{
    AVCodec *ctx = abcdk_avcodec_find(name,encode);

    if(ctx)
        return abcdk_avcodec_alloc(ctx);
    
    return NULL;
}

AVCodecContext *abcdk_avcodec_alloc3(enum AVCodecID id,int encode)
{
    AVCodec *ctx = abcdk_avcodec_find2(id,encode);

    if(ctx)
        return abcdk_avcodec_alloc(ctx);

    return NULL;
}

int abcdk_avcodec_open(AVCodecContext *ctx, AVDictionary **dict)
{
    int chk = -1;

    assert(ctx != NULL);
    assert(ctx->codec != NULL);

    /*如果是编码器，填写默认值。*/
    if (av_codec_is_encoder(ctx->codec))
    {
        if (dict)
        {
            if (ctx->codec_id == AV_CODEC_ID_H265)
                av_dict_set(dict, "x265-params", "bframes=0", 0);
            else if (ctx->codec_id == AV_CODEC_ID_H264)
                av_dict_set(dict, "x264opts", "bframes=0", 0);
        }
    }

    chk = avcodec_open2(ctx, NULL, dict);

    return chk;
}

int abcdk_avcodec_decode(AVCodecContext *ctx, AVFrame *out,const AVPacket *in)
{
    AVPacket tmp;
    int got = -1;
    int chk;

    assert(ctx != NULL && out != NULL);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)

    av_init_packet(&tmp);
    tmp.data = NULL;
    tmp.size = 0;

    /*No output.*/
    got = 0;

    if (ctx->codec->type == AVMEDIA_TYPE_VIDEO)
    {
        if (avcodec_decode_video2(ctx, out, &got, (in?in:&tmp)) < 0)
            return -1;
    }
    else if (ctx->codec->type == AVMEDIA_TYPE_AUDIO)
    {
        if (avcodec_decode_audio4(ctx, out, &got, (in?in:&tmp)) < 0)
            return -1;
    }
    else
    {
        return -2;
    }

    return got;
#else 

    chk = avcodec_send_packet(ctx, in);
    if(chk < 0)
        return -1;

    chk = avcodec_receive_frame(ctx,out);
    if(chk == AVERROR(EAGAIN) || chk == AVERROR_EOF)
        return 0;
    else if(chk < 0)
        return -1;

    return 1;

#endif

}

int abcdk_avcodec_encode(AVCodecContext *ctx, AVPacket *out, const AVFrame *in)
{
    int got = -1;
    int chk;

    assert(ctx != NULL && out != NULL);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    /*No output.*/
    got = 0;

    if (ctx->codec->type == AVMEDIA_TYPE_VIDEO)
    {
        if (avcodec_encode_video2(ctx, out, in, &got) != 0)
            return -1;
    }
    else if (ctx->codec->type == AVMEDIA_TYPE_AUDIO)
    {
        if (avcodec_encode_audio2(ctx, out, in, &got) != 0)
            return -1;
    }
    else
    {
        return -2;
    }

    return got;
#else 

    chk = avcodec_send_frame(ctx, in);
    if(chk < 0)
        return -1;
    
    chk = avcodec_receive_packet(ctx, out);
    if(chk == AVERROR(EAGAIN) || chk == AVERROR_EOF)
        return 0;
    else if(chk < 0)
        return -1;

    return 1;
#endif
}

void abcdk_avcodec_encode_video_fill_time_base(AVCodecContext *ctx, double fps)
{
#if 1

    /*-------------Copy from OpenCV----begin------------------*/

    int frame_rate = (int)(fps + 0.5);
    int frame_rate_base = 1;
    while (fabs(((double)frame_rate / frame_rate_base) - fps) > 0.001)
    {
        frame_rate_base *= 10;
        frame_rate = (int)(fps * frame_rate_base + 0.5);
    }

    ctx->time_base.den = frame_rate;
    ctx->time_base.num = frame_rate_base;

    /* adjust time base for supported framerates */
    if (ctx->codec && ctx->codec->supported_framerates)
    {
        const AVRational *p = ctx->codec->supported_framerates;
        AVRational req = {frame_rate, frame_rate_base};
        const AVRational *best = NULL;
        AVRational best_error = {INT_MAX, 1};
        for (; p->den != 0; p++)
        {
            AVRational error = av_sub_q(req, *p);
            if (error.num < 0)
                error.num *= -1;
            if (av_cmp_q(error, best_error) < 0)
            {
                best_error = error;
                best = p;
            }
        }

        if (best)
        {
            ctx->time_base.den = best->num;
            ctx->time_base.num = best->den;
        }
    }
    /*-------------Copy from OpenCV-----end---------------*/
#else
    ctx->time_base = (AVRational){1, fps};
#endif

    /*非常重要。*/
    ctx->framerate.den = ctx->time_base.num;
    ctx->framerate.num = ctx->time_base.den;
}

#pragma GCC diagnostic pop

#endif //AVCODEC_AVCODEC_H

