/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/jpeg.h"


__BEGIN_DECLS

#ifdef __cuda_cuda_h__

abcdk_torch_image_t *abcdk_cuda_jpeg_load(const char *src, CUcontext cuda_ctx)
{
    abcdk_torch_image_t *dst;
    abcdk_torch_jcodec_t *ctx;
    abcdk_torch_jcodec_param_t param = {0};
    int chk;

    assert(src != NULL && cuda_ctx != NULL);

    ctx = abcdk_cuda_jpeg_create(0, cuda_ctx);
    if(!ctx)
        return NULL;
    
    chk = abcdk_cuda_jpeg_start(ctx,&param);
    if(chk != 0)
    {
        abcdk_torch_jcodec_free(&ctx);
        return NULL;
    }    

    dst = abcdk_cuda_jpeg_decode_from_file(ctx,src);
    abcdk_torch_jcodec_free(&ctx);
    
    return dst;
}

#else //__cuda_cuda_h__

abcdk_torch_image_t *abcdk_cuda_jpeg_load(const char *src, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

#endif //__cuda_cuda_h__



__END_DECLS
