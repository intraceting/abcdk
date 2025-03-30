/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgcode.h"
#include "abcdk/torch/nvidia.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

int abcdk_torch_imgcode_save_cuda(const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_torch_jcodec_t *ctx;
    abcdk_torch_jcodec_param_t param = {0};
    int chk;

    assert(dst != NULL && src != NULL);

    assert(src->tag == ABCDK_TORCH_TAG_CUDA);

    ctx = abcdk_torch_jcodec_alloc_cuda(1);
    if(!ctx)
        return -1;

    param.quality = 95;

    chk = abcdk_torch_jcodec_start_cuda(ctx, &param);
    if (chk != 0)
    {
        abcdk_torch_jcodec_free_cuda(&ctx);
        return -2;
    }

    chk = abcdk_torch_jcodec_encode_to_file_cuda(ctx,dst,src);
    if(chk != 0)
    {
        abcdk_torch_jcodec_free_cuda(&ctx);
        return -3;
    }

    abcdk_torch_jcodec_free_cuda(&ctx);
    return 0;
}

abcdk_torch_image_t *abcdk_torch_imgcode_load_cuda(const char *src)
{
    abcdk_torch_image_t *dst;
    abcdk_torch_jcodec_t *ctx;
    abcdk_torch_jcodec_param_t param = {0};
    int chk;

    assert(src != NULL);

    ctx = abcdk_torch_jcodec_alloc_cuda(0);
    if(!ctx)
        return NULL;
    
    chk = abcdk_torch_jcodec_start_cuda(ctx,&param);
    if(chk != 0)
    {
        abcdk_torch_jcodec_free_cuda(&ctx);
        return NULL;
    }    

    dst = abcdk_torch_jcodec_decode_from_file_cuda(ctx,src);
    abcdk_torch_jcodec_free_cuda(&ctx);
    
    return dst;
}

#else //__cuda_cuda_h__

int abcdk_torch_imgcode_save_cuda(const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

abcdk_torch_image_t *abcdk_torch_imgcode_load_cuda(const char *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

#endif //__cuda_cuda_h__

__END_DECLS