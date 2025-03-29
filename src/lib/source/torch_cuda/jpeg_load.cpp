/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/jcodec.h"


__BEGIN_DECLS

#ifdef __cuda_cuda_h__

abcdk_torch_image_t *abcdk_torch_jcodec_load_cuda(const char *src)
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

abcdk_torch_image_t *abcdk_torch_jcodec_load_cuda(const char *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

#endif //__cuda_cuda_h__



__END_DECLS
