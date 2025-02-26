/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/packet.h"

static void _abcdk_media_packet_buffer_free_cb(void **ptr, int size)
{
    abcdk_heap_freep(ptr);
}

static int _abcdk_media_packet_buffer_alloc_cb(void **ptr, int size)
{
    *ptr = abcdk_heap_alloc(size);
    if (*ptr)
        return 0;

    return -1;
}

static void _abcdk_media_packet_clear(abcdk_media_packet_t *ctx)
{
    if (ctx->buffer_free_cb)
        ctx->buffer_free_cb(&ctx->buf_ptr, ctx->buf_size);

    ctx->data = NULL;
    ctx->size = -1;
    ctx->dts = -1;
    ctx->pts = -1;
}

void abcdk_media_packet_free(abcdk_media_packet_t **ctx)
{
    abcdk_media_packet_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    _abcdk_media_packet_clear(ctx_p);

    abcdk_heap_free(ctx_p);
}


abcdk_media_packet_t *abcdk_media_packet_alloc(uint32_t tag)
{
    abcdk_media_packet_t *ctx;

    assert(tag == ABCDK_MEDIA_TAG_HOST || tag == ABCDK_MEDIA_TAG_CUDA);

    ctx = (abcdk_media_packet_t *)abcdk_heap_alloc(sizeof(abcdk_media_packet_t));
    if (!ctx)
        return NULL;
    
    _abcdk_media_packet_clear(ctx);


    ctx->tag = tag;
    ctx->buffer_free_cb = _abcdk_media_packet_buffer_free_cb;
    ctx->buffer_alloc_cb = _abcdk_media_packet_buffer_alloc_cb;

    return ctx;
}


int abcdk_media_packet_reset(abcdk_media_packet_t *ctx, int size)
{
    int chk;

    assert(ctx != NULL && size > 0);
    assert(ctx->tag == ABCDK_MEDIA_TAG_HOST || ctx->tag == ABCDK_MEDIA_TAG_CUDA);

    if(ctx->size == size)
        return 0;

    _abcdk_media_packet_clear(ctx);

    chk = ctx->buffer_alloc_cb(&ctx->buf_ptr,ctx->buf_size = size);
    if (chk != 0)
        goto ERR;

    ctx->data = ctx->buf_ptr;
    ctx->size = ctx->buf_size;

    return 0;

ERR:

    _abcdk_media_packet_clear(ctx);

    return -1;
}


abcdk_media_packet_t *abcdk_media_packet_create(int size)
{
    abcdk_media_packet_t *ctx;
    int chk;

    assert(size > 0);

    ctx = abcdk_media_packet_alloc(ABCDK_MEDIA_TAG_HOST);
    if (!ctx)
        return NULL;

    chk = abcdk_media_packet_reset(ctx, size);
    if(chk != 0)
    {
        abcdk_media_packet_free(&ctx);
        return NULL;
    }

    return ctx;
}