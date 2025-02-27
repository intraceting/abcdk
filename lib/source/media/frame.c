/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/frame.h"

static void _abcdk_media_frame_private_free_cb(void **ctx)
{
    abcdk_media_image_free((abcdk_media_image_t **)ctx);
}

static int _abcdk_media_frame_image_upload_cb(void *ctx, const abcdk_media_image_t *src)
{
    abcdk_media_image_t *dst_p = NULL;
    int chk;

    dst_p = (abcdk_media_image_t *)ctx;

    chk = abcdk_media_image_reset(dst_p, src->width, src->height, src->pixfmt, 1);
    if (chk != 0)
        return -1;

    abcdk_media_image_copy(dst_p, src);

    return 0;
}

static int _abcdk_media_frame_image_download_cb(void *ctx, abcdk_media_image_t **dst)
{
    abcdk_media_image_t *src_p = NULL, *dst_p = NULL;

    src_p = (abcdk_media_image_t *)ctx;

    if (*dst)
        dst_p = (abcdk_media_image_t *)*dst;
    else
        dst_p = *dst = abcdk_media_image_create(src_p->width, src_p->height, src_p->pixfmt, 1);

    if (!dst_p)
        return -1;

    abcdk_media_image_copy(dst_p, src_p);

    return 0;
}

void abcdk_media_frame_free(abcdk_media_frame_t **ctx)
{
    abcdk_media_frame_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->private_ctx_free_cb)
        ctx_p->private_ctx_free_cb(&ctx_p->private_ctx);

    abcdk_heap_free(ctx_p);
}

abcdk_media_frame_t *abcdk_media_frame_alloc(uint32_t tag)
{
    abcdk_media_frame_t *ctx;

    assert(tag == ABCDK_MEDIA_TAG_HOST || tag == ABCDK_MEDIA_TAG_CUDA);

    ctx = (abcdk_media_frame_t *)abcdk_heap_alloc(sizeof(abcdk_media_frame_t));
    if (!ctx)
        return NULL;

    ctx->tag = tag;
    ctx->private_ctx = _abcdk_media_frame_private_free_cb;
    ctx->image_upload_cb = _abcdk_media_frame_image_upload_cb;
    ctx->image_download_cb = _abcdk_media_frame_image_download_cb;
    ctx->dts = -1;
    ctx->pts = -1;

    /*创建私有环境。*/
    ctx->private_ctx = abcdk_media_image_alloc(tag);
    if (!ctx->private_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_media_frame_free(&ctx);
    return NULL;
}

int abcdk_media_frame_image_upload(abcdk_media_frame_t *ctx, const abcdk_media_image_t *src)
{
    int chk;

    assert(ctx != NULL && src != NULL);

    chk = ctx->image_upload_cb(ctx->private_ctx, src);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_media_frame_image_download(abcdk_media_frame_t *ctx, abcdk_media_image_t **dst)
{
    int chk;

    assert(ctx != NULL && dst != NULL);

    chk = ctx->image_download_cb(ctx->private_ctx, dst);
    if (chk != 0)
        return -1;

    return 0;
}