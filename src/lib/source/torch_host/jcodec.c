/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/jcodec.h"


void abcdk_torch_jcodec_free_host(abcdk_torch_jcodec_t **ctx)
{
    abcdk_torch_jcodec_t *ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    abcdk_heap_free(ctx_p);
}

abcdk_torch_jcodec_t *abcdk_torch_jcodec_alloc_host(int encoder)
{
    abcdk_torch_jcodec_t *ctx;

    ctx = (abcdk_torch_jcodec_t *)abcdk_heap_alloc(sizeof(abcdk_torch_jcodec_t));
    if(!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_HOST;
    ctx->private_ctx = NULL;

    return ctx;
}

int abcdk_torch_jcodec_start_host(abcdk_torch_jcodec_t *ctx, abcdk_torch_jcodec_param_t *param)
{
    return -1;
}

abcdk_object_t *abcdk_torch_jcodec_encode_host(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src)
{
    return NULL;
}

int abcdk_torch_jcodec_encode_to_file_host(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src)
{
    return -1;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_host(abcdk_torch_jcodec_t *ctx, const void *src, int src_size)
{
    return NULL;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_from_file_host(abcdk_torch_jcodec_t *ctx, const void *src)
{
    return NULL;
}

int abcdk_torch_jcodec_save_host(const char *dst, const abcdk_torch_image_t *src)
{
    return -1;
}

abcdk_torch_image_t *abcdk_torch_jcodec_load_host(const char *src)
{
    return NULL;
}