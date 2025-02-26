/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/jcodec.h"


static void _abcdk_media_jcodec_private_free_cb(void **ctx, uint8_t encoder)
{

}


static void _abcdk_media_jcodec_clear(abcdk_media_jcodec_t *ctx)
{
    if (ctx->private_ctx_free_cb)
        ctx->private_ctx_free_cb(&ctx->private_ctx,ctx->encoder);
}

void abcdk_media_jcodec_free(abcdk_media_jcodec_t **ctx)
{
    abcdk_media_jcodec_t *ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;

    _abcdk_media_jcodec_clear(ctx);

    abcdk_heap_free(ctx_p);
}

abcdk_media_jcodec_t *abcdk_media_jcodec_alloc(uint32_t tag)
{
    abcdk_media_jcodec_t *ctx;

    assert(tag == ABCDK_MEDIA_TAG_HOST || tag == ABCDK_MEDIA_TAG_CUDA);

    ctx = (abcdk_media_jcodec_t *)abcdk_heap_alloc(sizeof(abcdk_media_jcodec_t));
    if(!ctx)
        return NULL;

    ctx->tag = tag;
    ctx->private_ctx = _abcdk_media_jcodec_private_free_cb;

    return ctx;
}