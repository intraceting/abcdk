/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/nvidia/context.h"

__BEGIN_DECLS

typedef struct _abcdk_cuda_context
{
    CUcontext impl_ctx;

}abcdk_cuda_context_t;

static void _abcdk_cuda_context_free_cb(void *userdata)
{
    abcdk_cuda_context_t *ctx = (abcdk_cuda_context_t*)userdata;

    cuCtxDestroy(ctx->impl_ctx);
}

abcdk_context_t *abcdk_cuda_context_create(int device, int flag)
{
    abcdk_context_t *ctx = NULL;
    abcdk_cuda_context_t *cu_ctx = NULL;
    CUdevice cuda_dev;
    CUresult chk;

    assert(device >=0);

    ctx = abcdk_context_alloc(sizeof(abcdk_cuda_context_t),_abcdk_cuda_context_free_cb);
    if(!ctx)
        return NULL;

    cu_ctx = (abcdk_cuda_context_t*)abcdk_context_get_userdata(ctx);

    chk = cuDeviceGet(&cuda_dev, device);
    if (chk != CUDA_SUCCESS)
        goto ERR;

    chk = cuCtxCreate(&cu_ctx->impl_ctx, flag, cuda_dev);
    if (chk != CUDA_SUCCESS)
        goto ERR;

    return ctx;

ERR:

    abcdk_context_unref(&ctx);

    return NULL;
}

static volatile int _abcdk_cuda_context_specific_init_status = 0;
static pthread_key_t _abcdk_cuda_context_specific_key = 0xFFFFFFFF;

static void _abcdk_cuda_context_specific_free(void *opaque)
{
    abcdk_context_t *ctx = (abcdk_context_t *)opaque;

    abcdk_context_unref(&ctx);
}

static int _abcdk_cuda_context_specific_init(void *opaque)
{
    pthread_key_t *key_p = (pthread_key_t*)opaque;

    pthread_key_create(key_p,_abcdk_cuda_context_specific_free);
    return 0;
}


int abcdk_cuda_context_setspecific(abcdk_context_t *ctx)
{
    abcdk_context_t *old_ctx = NULL;
    int chk;

    assert(ctx != NULL);

    /*注册KEY。*/
    chk = abcdk_once(&_abcdk_cuda_context_specific_init_status,_abcdk_cuda_context_specific_init,&_abcdk_cuda_context_specific_key);
    if(chk < 0)
        return -1;

    /*获取旧的并释放。*/
    old_ctx = (abcdk_context_t *)pthread_getspecific(_abcdk_cuda_context_specific_key);
    _abcdk_cuda_context_specific_free(old_ctx);

    /*绑定新的。*/
    chk = pthread_setspecific(_abcdk_cuda_context_specific_key,abcdk_context_refer(ctx));
    if(chk != 0)
    {
        abcdk_context_unref(&ctx);//unref
        return -1;
    }
    
    return 0;
}

abcdk_context_t *abcdk_cuda_context_getspecific()
{
    abcdk_context_t *old_ctx = NULL;
    int chk;

    /*注册KEY。*/
    chk = abcdk_once(&_abcdk_cuda_context_specific_init_status,_abcdk_cuda_context_specific_init,&_abcdk_cuda_context_specific_key);
    if(chk < 0)
        return NULL;

    /*获取旧的。*/
    old_ctx = (abcdk_context_t *)pthread_getspecific(_abcdk_cuda_context_specific_key);

    return old_ctx;
}

int abcdk_cuda_context_push(abcdk_context_t *ctx)
{
    abcdk_cuda_context_t *cu_ctx = NULL;
    CUresult chk;

    assert(ctx != NULL);

    cu_ctx = (abcdk_cuda_context_t*)abcdk_context_get_userdata(ctx);

    chk = cuCtxPushCurrent(cu_ctx->impl_ctx);
    if (chk != CUDA_SUCCESS)
        return -1;

    return 0;
}

int abcdk_cuda_context_pop(abcdk_context_t *ctx)
{
    abcdk_cuda_context_t *cu_ctx = NULL;
    CUcontext old_impl_ctx = NULL;
    CUresult chk;

    assert(ctx != NULL);

    cu_ctx = (abcdk_cuda_context_t*)abcdk_context_get_userdata(ctx);

    chk = cuCtxPopCurrent(&old_impl_ctx);
    if (chk != CUDA_SUCCESS)
        return -1;

    assert(cu_ctx->impl_ctx == old_impl_ctx);

    return 0;
}

__END_DECLS