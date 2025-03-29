/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/jcodec.h"


__BEGIN_DECLS

#ifdef __cuda_cuda_h__

int abcdk_torch_jcodec_save_cuda(const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_torch_jcodec_t *ctx;
    abcdk_torch_jcodec_param_t param = {0};
    int chk;

    assert(dst != NULL && src != NULL);

    assert(src->tag == ABCDK_TORCH_TAG_CUDA);

    ctx = abcdk_torch_jcodec_alloc_cuda(1);
    if(!ctx)
        return -1;

    param.quality = 99;

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

#else //__cuda_cuda_h__

int abcdk_torch_jcodec_save_cuda(const char *dst, const abcdk_torch_image_t *src, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

#endif //__cuda_cuda_h__


__END_DECLS