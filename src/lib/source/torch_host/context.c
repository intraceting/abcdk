/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/context.h"


void abcdk_torch_context_destroy_host(abcdk_torch_context_t **ctx)
{
    abcdk_torch_context_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    abcdk_heap_free(ctx_p);

}

static pthread_once_t _abcdk_torch_context_key_init_status_host = PTHREAD_ONCE_INIT;
static pthread_key_t _abcdk_torch_context_key_host = 0xFFFFFFFF;

static void _abcdk_torch_context_key_init_host()
{
    pthread_key_create(&_abcdk_torch_context_key_host,NULL);
}

abcdk_torch_context_t *abcdk_torch_context_create_host(int id, int flag)
{
    abcdk_torch_context_t *ctx;
    int chk;

    assert(id >= 0);

    /*初始化一次。*/
    chk = pthread_once(&_abcdk_torch_context_key_init_status_host, _abcdk_torch_context_key_init_host);
    assert(chk == 0);

    ctx = (abcdk_torch_context_t *)abcdk_heap_alloc(sizeof(abcdk_torch_context_t));
    if (!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_HOST;

    if (id >= 1)
        goto ERR;

    return ctx;

ERR:

    abcdk_torch_context_destroy_host(&ctx);

    return NULL;
}

int abcdk_torch_context_current_set_host(abcdk_torch_context_t *ctx)
{
    abcdk_torch_context_t *old_ctx = NULL;
    int chk;

    if (ctx)
    {
        assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

        old_ctx = (abcdk_torch_context_t *)pthread_getspecific(_abcdk_torch_context_key_host);
        if (old_ctx == ctx)
            return 0;

        ABCDK_ASSERT(old_ctx == NULL, TT("当前线程已经绑定运行环境，不能重复绑定。"));

        /*绑定到线程。*/
        chk = pthread_setspecific(_abcdk_torch_context_key_host, ctx);
        if (chk != 0)
            return -1;

    }
    else
    {
        old_ctx = (abcdk_torch_context_t *)pthread_getspecific(_abcdk_torch_context_key_host);
        if (!old_ctx)
            return 0;
        
        pthread_setspecific(_abcdk_torch_context_key_host, NULL); // unset.
    }

    return 0;
}

abcdk_torch_context_t *abcdk_torch_context_current_get_host()
{
    abcdk_torch_context_t *old_ctx = NULL;

    old_ctx = (abcdk_torch_context_t *)pthread_getspecific(_abcdk_torch_context_key_host);
    ABCDK_ASSERT(old_ctx != NULL, TT("当前线程尚未绑定运行环境。"));

    return old_ctx;
}