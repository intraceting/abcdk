/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/vcodec.h"


void abcdk_torch_vcodec_free_host(abcdk_torch_vcodec_t **ctx)
{
    abcdk_torch_vcodec_t *ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;
    
    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    abcdk_heap_free(ctx_p);
}

abcdk_torch_vcodec_t *abcdk_torch_vcodec_alloc_host(int encoder)
{
    abcdk_torch_vcodec_t *ctx;

    ctx = (abcdk_torch_vcodec_t *)abcdk_heap_alloc(sizeof(abcdk_torch_vcodec_t));
    if(!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_HOST;
    ctx->private_ctx = NULL;

    return ctx;
}

int abcdk_torch_vcodec_start_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_vcodec_param_t *param)
{
    return -1;
}

int abcdk_torch_vcodec_encode_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src)
{
    return -1;
}

int abcdk_torch_vcodec_decode_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const abcdk_torch_packet_t *src)
{
    return -1;
}

#ifdef AVCODEC_AVCODEC_H

int abcdk_torch_vcodec_encode_to_ffmpeg_host(abcdk_torch_vcodec_t *ctx, AVPacket **dst, const abcdk_torch_frame_t *src)
{
    return -1;
}

int abcdk_torch_vcodec_decode_from_ffmpeg_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const AVPacket *src)
{
    return -1;
}

#else //AVCODEC_AVCODEC_H

int abcdk_torch_vcodec_encode_to_ffmpeg_host(abcdk_torch_vcodec_t *ctx, AVPacket **dst, const abcdk_torch_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return -1;
}

int abcdk_torch_vcodec_decode_from_ffmpeg_host(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const AVPacket *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return -1;
}

#endif //AVCODEC_AVCODEC_H