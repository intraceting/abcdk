/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/torch/context.h"
#include "abcdk/torch/nvidia.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

void abcdk_torch_context_destroy_cuda(abcdk_torch_context_t **ctx)
{
    abcdk_torch_context_t *ctx_p;
    CUcontext cu_ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (CUcontext)ctx_p->private_ctx;

    cuCtxDestroy(cu_ctx_p);
    abcdk_heap_free(ctx_p);

}

static pthread_once_t _abcdk_torch_context_key_init_status_cuda = PTHREAD_ONCE_INIT;
static pthread_key_t _abcdk_torch_context_key_cuda = 0xFFFFFFFF;

static void _abcdk_torch_context_key_init_cuda()
{
    pthread_key_create(&_abcdk_torch_context_key_cuda,NULL);
}

abcdk_torch_context_t *abcdk_torch_context_create_cuda(int id, int flag)
{    
    abcdk_torch_context_t *ctx;
    CUdevice dev_ctx;
    CUresult cu_chk;
    int chk;

    assert(id >= 0);

    /*初始化一次。*/
    chk = pthread_once(&_abcdk_torch_context_key_init_status_cuda,_abcdk_torch_context_key_init_cuda);
    assert(chk == 0);

    ctx = (abcdk_torch_context_t *)abcdk_heap_alloc(sizeof(abcdk_torch_context_t));
    if(!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_CUDA;

    cu_chk = cuDeviceGet(&dev_ctx, id);
    if (cu_chk != CUDA_SUCCESS)
        goto ERR;

    cu_chk = cuCtxCreate((CUcontext*)&ctx->private_ctx, flag, dev_ctx);
    if (cu_chk != CUDA_SUCCESS)
        goto ERR;

    return ctx;

ERR:

    abcdk_torch_context_destroy_cuda(&ctx);

    return NULL;
}

int abcdk_torch_context_current_set_cuda(abcdk_torch_context_t *ctx)
{
    abcdk_torch_context_t *old_ctx = NULL;
    CUcontext cu_old_ctx = NULL;
    CUresult cu_chk;
    int chk;

    if (ctx)
    {
        old_ctx = (abcdk_torch_context_t *)pthread_getspecific(_abcdk_torch_context_key_cuda);
        if (old_ctx == ctx)
            return 0;

        ABCDK_ASSERT(old_ctx == NULL, TT("当前线程已经绑定运行环境，不能重复绑定。"));

        /*绑定到线程。*/
        chk = pthread_setspecific(_abcdk_torch_context_key_cuda, ctx);
        if (chk != 0)
            return -1;

        /*绑定到设备。*/
        cu_chk = cuCtxPushCurrent((CUcontext)ctx->private_ctx);
        if (cu_chk != CUDA_SUCCESS)
        {
            pthread_setspecific(_abcdk_torch_context_key_cuda, NULL); // unset.
            return -1;
        }
    }
    else
    {
        old_ctx = (abcdk_torch_context_t *)pthread_getspecific(_abcdk_torch_context_key_cuda);
        if (!old_ctx)
            return 0;

        /*解除设备绑定。*/
        cuCtxPopCurrent(&cu_old_ctx);

        ABCDK_ASSERT(old_ctx->private_ctx == cu_old_ctx,TT("线程运行环境不能交叉绑定，或其它操作没有配对。"));
        
        pthread_setspecific(_abcdk_torch_context_key_cuda, NULL); // unset.
    }

    return 0;
}

abcdk_torch_context_t *abcdk_torch_context_current_get_cuda()
{
    abcdk_torch_context_t *old_ctx = NULL;

    old_ctx = (abcdk_torch_context_t *)pthread_getspecific(_abcdk_torch_context_key_cuda);
    ABCDK_ASSERT(old_ctx != NULL, TT("当前线程未绑定运行环境。"));

    return old_ctx;
}


#else // __cuda_cuda_h__

void abcdk_torch_context_destroy_cuda(abcdk_torch_context_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return ;
}

abcdk_torch_context_t *abcdk_torch_context_create_cuda(int device, int flag)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_torch_context_current_set_cuda(abcdk_torch_context_t *ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

abcdk_torch_context_t *abcdk_torch_context_current_get_cuda()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

#endif //__cuda_cuda_h__


__END_DECLS