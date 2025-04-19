/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/vcodec.h"
#include "abcdk/ffmpeg/avcodec.h"

#ifdef AVCODEC_AVCODEC_H

/** 视频编/解码器。*/
typedef struct _abcdk_torch_vcodec_host
{
    /**编码器。!0 是，0 否。*/
    int encoder;

    /**编/解码器环境。*/
    AVCodecContext *ff_ctx;

    /**编码时间轴。*/
    int64_t ff_en_pts;

    /*临时变量。*/

    AVPacket *ff_tmp_pkt;
    AVFrame *ff_tmp_fae;

} abcdk_torch_vcodec_host_t;

void abcdk_torch_vcodec_free_host(abcdk_torch_vcodec_t **ctx)
{
    abcdk_torch_vcodec_t *ctx_p;
    abcdk_torch_vcodec_host_t *ht_ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;
    
    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk_torch_vcodec_host_t *)ctx_p->private_ctx;

    if(ht_ctx_p)
    {
        abcdk_avcodec_free(&ht_ctx_p->ff_ctx);
        av_packet_free(&ht_ctx_p->ff_tmp_pkt);
        av_frame_free(&ht_ctx_p->ff_tmp_fae);
    }

    abcdk_heap_free(ctx_p->private_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_torch_vcodec_t *abcdk_torch_vcodec_alloc_host(int encoder)
{
    abcdk_torch_vcodec_t *ctx;
    abcdk_torch_vcodec_host_t *ht_ctx_p;

    ctx = (abcdk_torch_vcodec_t *)abcdk_heap_alloc(sizeof(abcdk_torch_vcodec_t));
    if(!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_HOST;

    ctx->private_ctx = abcdk_heap_alloc(sizeof(abcdk_torch_vcodec_host_t));
    if(!ctx->private_ctx)
        goto ERR;

    ht_ctx_p = (abcdk_torch_vcodec_host_t *)ctx->private_ctx;

    ht_ctx_p->encoder = encoder;

    ht_ctx_p->ff_tmp_pkt = av_packet_alloc();
    ht_ctx_p->ff_tmp_fae = av_frame_alloc();

    return ctx;

ERR:

    abcdk_torch_vcodec_free_host(&ctx);
    return NULL;
}

int abcdk_torch_vcodec_start_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_vcodec_param_t *param)
{
    abcdk_torch_vcodec_host_t *ht_ctx_p;
    enum AVCodecID ff_fmt;
    int chk;

    assert(ctx != NULL && param != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk_torch_vcodec_host_t *)ctx->private_ctx;

    ff_fmt = (enum AVCodecID)abcdk_torch_vcodec_convert_to_ffmpeg(param->format);
    if(ff_fmt <= AV_CODEC_ID_NONE)
        return -1;

    ht_ctx_p->ff_ctx = abcdk_avcodec_alloc3(ff_fmt,ht_ctx_p->encoder);
    if(!ht_ctx_p->ff_ctx)
        return -1;

    if (ht_ctx_p->encoder)
    {
        ht_ctx_p->ff_ctx->time_base = av_make_q(param->fps_n,param->fps_d);
        ht_ctx_p->ff_ctx->framerate.den = param->fps_n;
        ht_ctx_p->ff_ctx->framerate.num = param->fps_d;
        ht_ctx_p->ff_ctx->pkt_timebase.num = param->fps_n;
        ht_ctx_p->ff_ctx->pkt_timebase.den = param->fps_d;

        ht_ctx_p->ff_ctx->width = param->width;
        ht_ctx_p->ff_ctx->height = param->height;
        ht_ctx_p->ff_ctx->gop_size = param->fps_n/param->fps_d;
        ht_ctx_p->ff_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        //ht_ctx_p->ff_ctx->pix_fmt = (ht_ctx_p->ff_ctx->codec->pix_fmts ? ht_ctx_p->ff_ctx->codec->pix_fmts[0] : AV_PIX_FMT_YUV420P);

        /*No b frame.*/
        ht_ctx_p->ff_ctx->max_b_frames = 0;

        /*设置全局头部。*/
        ht_ctx_p->ff_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        chk = abcdk_avcodec_open(ht_ctx_p->ff_ctx, NULL);
        if(chk < 0)
            return -1;

        param->ext_data = ht_ctx_p->ff_ctx->extradata;
        param->ext_size = ht_ctx_p->ff_ctx->extradata_size;

        ht_ctx_p->ff_en_pts = 0;//set 0
    }
    else
    {
        ht_ctx_p->ff_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
        ht_ctx_p->ff_ctx->codec_id = ff_fmt;
        ht_ctx_p->ff_ctx->width = param->width;
        ht_ctx_p->ff_ctx->height = param->height;
        ht_ctx_p->ff_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

        av_freep(&ht_ctx_p->ff_ctx->extradata);

        ht_ctx_p->ff_ctx->extradata_size = param->ext_size;
        ht_ctx_p->ff_ctx->extradata = (uint8_t *)av_mallocz(param->ext_size + AV_INPUT_BUFFER_PADDING_SIZE);

        memcpy(ht_ctx_p->ff_ctx->extradata, param->ext_data, param->ext_size);

        chk = abcdk_avcodec_open(ht_ctx_p->ff_ctx, NULL);
        if(chk < 0)
            return -1;
    }

    return 0;
}

int abcdk_torch_vcodec_encode_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src)
{
    abcdk_torch_vcodec_host_t *ht_ctx_p;
    abcdk_torch_frame_t *tmp_src;
    int chk;

    assert(ctx != NULL && dst != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk_torch_vcodec_host_t *)ctx->private_ctx;

    if (src)
    {
        assert(src->img->tag == ABCDK_TORCH_TAG_HOST);

        if (src->img->pixfmt != ABCDK_TORCH_PIXFMT_YUV420P)
        {
            tmp_src = abcdk_torch_frame_create_host(src->img->width, src->img->height, ABCDK_TORCH_PIXFMT_YUV420P, 1);
            if (!tmp_src)
                return -1;

            /*转换格式。*/
            chk = abcdk_torch_image_convert_host(tmp_src->img, src->img);

            if (chk == 0)
            {
                /*复制其它参数。*/
                tmp_src->dts = src->dts;
                tmp_src->pts = src->pts;

                chk = abcdk_torch_vcodec_encode_host(ctx, dst, tmp_src);
            }

            abcdk_torch_frame_free(&tmp_src);

            return chk;
        }

        av_frame_unref(ht_ctx_p->ff_tmp_fae);

        for (int i = 0; i < 4; i++)
        {
            if(!src->img->data[i])
                break;

            ht_ctx_p->ff_tmp_fae->data[i] = src->img->data[i];
            ht_ctx_p->ff_tmp_fae->linesize[i] = src->img->stride[i];
        }

        ht_ctx_p->ff_tmp_fae->width = src->img->width;
        ht_ctx_p->ff_tmp_fae->height = src->img->height;
        ht_ctx_p->ff_tmp_fae->format = (int)AV_PIX_FMT_YUV420P;

        ht_ctx_p->ff_tmp_fae->pts = ++ht_ctx_p->ff_en_pts; // 递增。
        ht_ctx_p->ff_tmp_fae->pkt_dts = (int64_t)AV_NOPTS_VALUE;
        ht_ctx_p->ff_tmp_fae->pkt_pts = (int64_t)AV_NOPTS_VALUE;

        /*下面设置会使编码器自行决定帧类型。*/
        ht_ctx_p->ff_tmp_fae->key_frame = 0;
        ht_ctx_p->ff_tmp_fae->pict_type = AV_PICTURE_TYPE_NONE;
    }

    av_packet_unref(ht_ctx_p->ff_tmp_pkt);
    chk = abcdk_avcodec_encode(ht_ctx_p->ff_ctx, ht_ctx_p->ff_tmp_pkt, (src ? ht_ctx_p->ff_tmp_fae : NULL));
    if(chk < 0 )
        return -1;
    else if(chk == 0)
        return 0;

    chk = abcdk_torch_packet_reset(dst, ht_ctx_p->ff_tmp_pkt->size);
    if (chk != 0)
        return -1;

    memcpy((*dst)->data, ht_ctx_p->ff_tmp_pkt->data, ht_ctx_p->ff_tmp_pkt->size);

    return 1;
}

int abcdk_torch_vcodec_decode_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const abcdk_torch_packet_t *src)
{
    abcdk_torch_vcodec_host_t *ht_ctx_p;
    abcdk_torch_frame_t *tmp_src;
    int chk;

    assert(ctx != NULL && dst != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk_torch_vcodec_host_t *)ctx->private_ctx;

    if(src)
    {
        av_packet_unref(ht_ctx_p->ff_tmp_pkt);

        ht_ctx_p->ff_tmp_pkt->data = src->data;
        ht_ctx_p->ff_tmp_pkt->size = src->size;
        ht_ctx_p->ff_tmp_pkt->pts = src->pts;
    }

    av_frame_unref(ht_ctx_p->ff_tmp_fae);
    chk = abcdk_avcodec_decode(ht_ctx_p->ff_ctx, ht_ctx_p->ff_tmp_fae, (src ? ht_ctx_p->ff_tmp_pkt : NULL));
    if(chk < 0 )
        return -1;
    else if(chk == 0)
        return 0;

    chk = abcdk_torch_frame_reset_host(dst, ht_ctx_p->ff_tmp_fae->width, ht_ctx_p->ff_tmp_fae->height, abcdk_torch_pixfmt_convert_from_ffmpeg(ht_ctx_p->ff_tmp_fae->format), 1);
    if (chk != 0)
        return -1;

    for (int i = 0; i < 4; i++)
    {
        if (!ht_ctx_p->ff_tmp_fae->data[i])
            break;

        abcdk_torch_image_copy_plane_host((*dst)->img, i, ht_ctx_p->ff_tmp_fae->data[i], ht_ctx_p->ff_tmp_fae->linesize[i]);
    }

    (*dst)->pts = ht_ctx_p->ff_tmp_fae->pts;//bind PTS

    return 1;
}

int abcdk_torch_vcodec_encode_to_ffmpeg_host(abcdk_torch_vcodec_t *ctx, AVPacket **dst, const abcdk_torch_frame_t *src)
{
    abcdk_torch_packet_t *tmp_dst = NULL;
    AVPacket *dst_p;
    int chk;

    assert(ctx != NULL && dst != NULL);

    chk = abcdk_torch_vcodec_encode_host(ctx, &tmp_dst, src);
    if (chk > 0)
    {
        dst_p = *dst;

        if (dst_p)
            av_packet_unref(dst_p);
        else
            dst_p = *dst = av_packet_alloc();

        if (!dst_p)
        {
            abcdk_torch_packet_free(&tmp_dst);
            return -1;
        }

        av_new_packet(dst_p, tmp_dst->size);
        memcpy(dst_p->data, tmp_dst->data, tmp_dst->size);
    }

    abcdk_torch_packet_free(&tmp_dst);

    return chk;
}

int abcdk_torch_vcodec_decode_from_ffmpeg_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const AVPacket *src)
{
    abcdk_torch_packet_t tmp_src = {0};

    assert(ctx != NULL && dst != NULL);

    if (src)
    {
        tmp_src.data = src->data;
        tmp_src.size = src->size;
        tmp_src.pts = src->pts;
    }

    return abcdk_torch_vcodec_decode_host(ctx, dst, (src ? &tmp_src : NULL));
}

#else //AVCODEC_AVCODEC_H

void abcdk_torch_vcodec_free_host(abcdk_torch_vcodec_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return ;
}

abcdk_torch_vcodec_t *abcdk_torch_vcodec_alloc_host(int encoder)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return NULL;
}

int abcdk_torch_vcodec_encode_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return -1;
}

int abcdk_torch_vcodec_decode_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const abcdk_torch_packet_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return -1;
}

#endif //AVCODEC_AVCODEC_H