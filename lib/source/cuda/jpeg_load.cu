/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/jpeg.h"

#ifdef __cuda_cuda_h__

abcdk_media_frame_t *abcdk_cuda_jpeg_load(const char *src, CUcontext cuda_ctx)
{
    abcdk_media_frame_t *dst;
    abcdk_cuda_jpeg_t *ctx;

    assert(src != NULL && cuda_ctx != NULL);

    ctx = abcdk_cuda_jpeg_create(0,NULL,cuda_ctx);
    if(!ctx)
        return NULL;

    dst = abcdk_cuda_jpeg_decode_from_file(ctx,src);
    abcdk_cuda_jpeg_destroy(&ctx);
    
    return dst;
}

#else //__cuda_cuda_h__

abcdk_media_frame_t *abcdk_cuda_jpeg_load(const char *src, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

#endif //__cuda_cuda_h__
