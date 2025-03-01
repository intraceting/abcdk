/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/frame.h"


void abcdk_media_frame_free(abcdk_media_frame_t **ctx)
{
    abcdk_media_frame_t *ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_media_image_free(&ctx_p->img);
    abcdk_heap_free(ctx_p);
}


abcdk_media_frame_t *abcdk_media_frame_alloc()
{
    abcdk_media_frame_t *ctx;

    ctx = (abcdk_media_frame_t*)abcdk_heap_alloc(sizeof(abcdk_media_frame_t));
    if(!ctx)
        return NULL;

    ctx->img = NULL;
    ctx->dts = -1;
    ctx->pts = -1;

    return ctx;
}