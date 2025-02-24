/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/opencv/stitcher.h"
#include "stitcher.hxx"

/*简单的全景拼接。*/
struct _abcdk_stitcher
{
#ifdef OPENCV_CORE_HPP
    abcdk::opencv::stitcher *opencv_ctx;
#endif // OPENCV_CORE_HPP
}; // abcdk_stitcher_t;

void abcdk_stitcher_destroy(abcdk_stitcher_t **ctx)
{
    abcdk_stitcher_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

#ifdef OPENCV_CORE_HPP
    delete ctx_p->opencv_ctx;
#endif // OPENCV_CORE_HPP

    abcdk_heap_free(ctx_p);
}

abcdk_stitcher_t *abcdk_stitcher_create()
{
    abcdk_stitcher_t *ctx;

    ctx = (abcdk_stitcher_t*)abcdk_heap_alloc(sizeof(abcdk_stitcher_t));
    if(!ctx)
        return NULL;

#ifdef OPENCV_CORE_HPP
    ctx->opencv_ctx = new abcdk::opencv::stitcher();
    if(!ctx->opencv_ctx)
        goto ERR;
#else // OPENCV_CORE_HPP
    abcdk_trace_printf(LOG_WARNING, "当前环境未包含OpenCV工具，无法创建对象。");
    goto ERR;
#endif // OPENCV_CORE_HPP

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

#ifdef OPENCV_CORE_HPP

    chk = abcdk::opencv::stitcher::Dump(out_data,*ctx->opencv_ctx,magic);
    if(chk != 0)
        return NULL;

    out = abcdk_object_copyfrom(out_data.c_str(),out_data.length());
    if(!out)
        return NULL;
#endif // OPENCV_CORE_HPP

    return out;
}

int abcdk_stitcher_metadata_load(abcdk_stitcher_t *ctx, const char *magic, const char *data)
{
    int chk;

    assert(ctx != NULL && magic != NULL && data != NULL);

#ifdef OPENCV_CORE_HPP

    chk = abcdk::opencv::stitcher::Load(data,*ctx->opencv_ctx,magic);
    if(chk == 0)
        return 0;
    else if (chk == -127)
        return -127;

#endif // OPENCV_CORE_HPP

    return -1;
}

