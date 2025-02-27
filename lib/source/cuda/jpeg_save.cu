/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/jpeg.h"

#ifdef __cuda_cuda_h__

int abcdk_cuda_jpeg_save(const char *dst, const abcdk_media_image_t *src, CUcontext cuda_ctx)
{
    abcdk_media_jcodec_t *ctx;
    abcdk_media_jcodec_param_t param = {0};
    int chk;

    assert(dst != NULL && src != NULL && cuda_ctx != NULL);

    ctx = abcdk_cuda_jpeg_create(1,cuda_ctx);
    if(!ctx)
        return -1;

    param.quality = 99;

    chk = abcdk_cuda_jpeg_start(ctx, &param);
    if (chk != 0)
    {
        abcdk_media_jcodec_free(&ctx);
        return -2;
    }

    chk = abcdk_cuda_jpeg_encode_to_file(ctx,dst,src);
    if(chk != 0)
    {
        abcdk_media_jcodec_free(&ctx);
        return -3;
    }

    abcdk_media_jcodec_free(&ctx);
    return 0;
}

#else //__cuda_cuda_h__

int abcdk_cuda_jpeg_save(const char *dst, const abcdk_media_image_t *src, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

#endif //__cuda_cuda_h__
