/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/opencv/stitcher.h"
#include "stitcher_general.hxx"

#ifdef OPENCV_CORE_HPP

/**简单的全景拼接引擎。*/
struct _abcdk_stitcher
{
    /**/
    abcdk::opencv::stitcher_general *impl_ctx;

}; // abcdk_stitcher_t;

void abcdk_stitcher_destroy(abcdk_stitcher_t **ctx)
{
    abcdk_stitcher_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    delete ctx_p->impl_ctx;

    abcdk_heap_free(ctx_p);
}

abcdk_stitcher_t *abcdk_stitcher_create()
{
    abcdk_stitcher_t *ctx;

    ctx = (abcdk_stitcher_t*)abcdk_heap_alloc(sizeof(abcdk_stitcher_t));
    if(!ctx)
        return NULL;

    ctx->impl_ctx = new abcdk::opencv::stitcher_general();
    if(!ctx->impl_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_stitcher_destroy(&ctx);
    return NULL;
}

abcdk_object_t *abcdk_stitcher_metadata_dump(abcdk_stitcher_t *ctx, const char *magic)
{
    abcdk_object_t *out = NULL;
    std::string out_data;
    int chk;

    assert(ctx != NULL && magic != NULL);

    chk = abcdk::opencv::stitcher_general::Dump(out_data,*ctx->impl_ctx,magic);
    if(chk != 0)
        return NULL;

    out = abcdk_object_copyfrom(out_data.c_str(),out_data.length());
    if(!out)
        return NULL;


    return out;
}

int abcdk_stitcher_metadata_load(abcdk_stitcher_t *ctx, const char *magic, const char *data)
{
    int chk;

    assert(ctx != NULL && magic != NULL && data != NULL);

    chk = abcdk::opencv::stitcher_general::Load(data,*ctx->impl_ctx,magic);
    if(chk == 0)
        return 0;
    else if (chk == -127)
        return -127;


    return -1;
}

#else //OPENCV_CORE_HPP

void abcdk_stitcher_destroy(abcdk_stitcher_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
}

abcdk_stitcher_t *abcdk_stitcher_create()
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

abcdk_object_t *abcdk_stitcher_metadata_dump(abcdk_stitcher_t *ctx, const char *magic)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

int abcdk_stitcher_metadata_load(abcdk_stitcher_t *ctx, const char *magic, const char *data)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return -1;
}

#endif // OPENCV_CORE_HPP

