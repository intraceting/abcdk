/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/context.h"

/**简单的上下文环境。 */
struct _abcdk_context 
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_CONTEXT_MAGIC 123456789

    /** 引用计数器。*/
    volatile int refcount;

     /**同步锁。*/
    abcdk_rwlock_t *locker_ctx;

    /** 用户环境指针。*/
    abcdk_object_t *userdata;

    /** 用户环境销毁函数。*/
    void (*userdata_free_cb)(void *userdata);
 
};//abcdk_context_t

void abcdk_context_unref(abcdk_context_t **ctx)
{
    abcdk_context_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->magic == ABCDK_CONTEXT_MAGIC);

    if (abcdk_atomic_fetch_and_add(&ctx_p->refcount, -1) != 1)
        return;

    assert(ctx_p->refcount == 0);

    ctx_p->magic = 0xcccccccc;

    if(ctx_p->userdata_free_cb)
        ctx_p->userdata_free_cb(ctx_p->userdata->pptrs[0]);

    abcdk_rwlock_destroy(&ctx_p->locker_ctx);
    abcdk_object_unref(&ctx_p->userdata);
    abcdk_heap_free(ctx_p);
}

abcdk_context_t *abcdk_context_refer(abcdk_context_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_context_t *abcdk_context_alloc(size_t userdata, void (*free_cb)(void *userdata))
{
    abcdk_context_t *ctx;

    ctx = (abcdk_context_t *)abcdk_heap_alloc(sizeof(abcdk_context_t));
    if (!ctx)
        return NULL;

    ctx->magic = ABCDK_CONTEXT_MAGIC;
    ctx->refcount = 1;

    ctx->locker_ctx = abcdk_rwlock_create();
    if(!ctx->locker_ctx)
        goto ERR;

    ctx->userdata = abcdk_object_alloc2(userdata);
    if(!ctx->userdata)
        goto ERR;

    ctx->userdata_free_cb = free_cb;

    return ctx;

ERR:

    abcdk_context_unref(&ctx);
    return NULL;
}

void *abcdk_context_get_userdata(abcdk_context_t *ctx)
{
    assert(ctx != NULL);

    return ctx->userdata->pptrs[0];
}

void abcdk_context_rdlock(abcdk_context_t *ctx)
{
    assert(ctx != NULL);

    abcdk_rwlock_rdlock(ctx->locker_ctx,1);
}


void abcdk_context_wrlock(abcdk_context_t *ctx)
{
    assert(ctx != NULL);

    abcdk_rwlock_wrlock(ctx->locker_ctx,1);
}


int abcdk_context_unlock(abcdk_context_t *ctx,int exitcode)
{
    assert(ctx != NULL);

    abcdk_rwlock_unlock(ctx->locker_ctx);

    return exitcode;
}
