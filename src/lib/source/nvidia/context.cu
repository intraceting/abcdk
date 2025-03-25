/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/nvidia/context.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

void abcdk_cuda_ctx_destroy(CUcontext *ctx)
{
    CUcontext ctx_p = NULL;

    ctx_p = *ctx;
    *ctx = NULL;

    cuCtxDestroy(ctx_p);
}

static pthread_once_t _abcdk_cuda_ctx_key_init_status = PTHREAD_ONCE_INIT;
static pthread_key_t _abcdk_cuda_ctx_key = 0xFFFFFFFF;

static void _abcdk_cuda_ctx_key_init()
{
    pthread_key_create(&_abcdk_cuda_ctx_key,NULL);
}

CUcontext abcdk_cuda_ctx_create(int device, int flag)
{    
    CUcontext ctx = NULL;
    CUdevice dev_ctx;
    CUresult cu_chk;
    int chk;

    assert(device >= 0);

    /*初始化一次。*/
    chk = pthread_once(&_abcdk_cuda_ctx_key_init_status,_abcdk_cuda_ctx_key_init);
    assert(chk == 0);

    cu_chk = cuDeviceGet(&dev_ctx, device);
    if (cu_chk != CUDA_SUCCESS)
        goto ERR;

    cu_chk = cuCtxCreate(&ctx, flag, dev_ctx);
    if (cu_chk != CUDA_SUCCESS)
        goto ERR;

    return ctx;

ERR:

    abcdk_cuda_ctx_destroy(&ctx);

    return NULL;
}

int abcdk_cuda_ctx_push(CUcontext ctx)
{
    CUresult cu_chk;

    assert(ctx != NULL);

    /*绑定到设备。*/
    cu_chk = cuCtxPushCurrent(ctx);
    if (cu_chk != CUDA_SUCCESS)
        return -1;
    
    return 0;
}

CUcontext abcdk_cuda_ctx_pop()
{
    CUcontext old_ctx = NULL;

    /*解除设备绑定。*/
    cuCtxPopCurrent(&old_ctx);

    return old_ctx;
}

int abcdk_cuda_ctx_setspecific(CUcontext ctx)
{
    int chk;

    /*绑定到线程。*/
    chk = pthread_setspecific(_abcdk_cuda_ctx_key, ctx);
    if (chk != 0)
        return -1;
    
    return 0;
}

CUcontext abcdk_cuda_ctx_getspecific()
{
    CUcontext old_ctx = NULL;

    old_ctx = (CUcontext)pthread_getspecific(_abcdk_cuda_ctx_key);
    ABCDK_ASSERT(old_ctx != NULL, TT("当前线程尚未绑定CUDA环境。"));

    return old_ctx;
}



#else // __cuda_cuda_h__

void abcdk_cuda_ctxt_destroy(CUcontext *ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return ;
}

CUcontext abcdk_cuda_ctxt_create(int device, int flag)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_cuda_ctx_push(CUcontext ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

CUcontext abcdk_cuda_ctx_pop()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_cuda_ctx_setspecific()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

CUcontext abcdk_cuda_ctx_getspecific()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

#endif //__cuda_cuda_h__


__END_DECLS