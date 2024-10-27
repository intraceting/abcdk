/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/rwlock.h"

/**
 * 读写锁。
*/
struct _abcdk_rwlock
{
    pthread_rwlock_t rwlock;
    pthread_rwlockattr_t rwattr;

} ;//abcdk_rwlock_t;

void abcdk_rwlock_destroy(abcdk_rwlock_t **ctx)
{
    abcdk_rwlock_t *ctx_p = NULL;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    pthread_rwlock_destroy(&ctx_p->rwlock);
    abcdk_heap_free(ctx_p);
    
}

abcdk_rwlock_t *abcdk_rwlock_create()
{
    abcdk_rwlock_t *ctx;

    ctx = (abcdk_rwlock_t *)abcdk_heap_alloc(sizeof(abcdk_rwlock_t));
    if(!ctx)
        return NULL;

    pthread_rwlockattr_init(&ctx->rwattr);
    pthread_rwlockattr_setpshared(&ctx->rwattr, PTHREAD_PROCESS_PRIVATE);

    pthread_rwlock_init(&ctx->rwlock,&ctx->rwattr);

    return ctx;
}   

int abcdk_rwlock_rdlock(abcdk_rwlock_t *ctx, int block)
{
    int err = -1;

    assert(ctx);

    if(block)
        err = pthread_rwlock_rdlock(&ctx->rwlock);
    else 
        err = pthread_rwlock_tryrdlock(&ctx->rwlock);

    return err;
}

int abcdk_rwlock_wrlock(abcdk_rwlock_t *ctx, int block)
{
    int err = -1;

    assert(ctx);

    if(block)
        err = pthread_rwlock_wrlock(&ctx->rwlock);
    else 
        err = pthread_rwlock_trywrlock(&ctx->rwlock);

    return err;
}

int abcdk_rwlock_unlock(abcdk_rwlock_t* ctx)
{
    int err = -1;

    assert(ctx);

    err = pthread_rwlock_unlock(&ctx->rwlock);
    
    return err;
}
