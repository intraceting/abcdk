/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/jpeg.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

int abcdk_cuda_jpeg_save(const char *dst, const AVFrame *src, CUcontext cuda_ctx)
{
    abcdk_cuda_jpeg_t *ctx;
    int chk;

    assert(dst != NULL && src != NULL && cuda_ctx != NULL);

    ctx = abcdk_cuda_jpeg_create(1,NULL,cuda_ctx);
    if(!ctx)
        return -1;

    chk = abcdk_cuda_jpeg_encode_to_file(ctx,dst,src);
    if(chk != 0)
    {
        abcdk_cuda_jpeg_destroy(&ctx);
        return -1;
    }

    abcdk_cuda_jpeg_destroy(&ctx);
    return 0;
}

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__
