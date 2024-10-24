/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/spinlock.h"

/**
 * 自旋锁。
*/
struct _abcdk_spinlock
{
    pthread_spinlock_t spinlock;

} ;//abcdk_spinlock_t;

void abcdk_spinlock_destroy(abcdk_spinlock_t **ctx)
{
    abcdk_spinlock_t *ctx_p = NULL;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    pthread_spin_destroy(&ctx_p->spinlock);
    abcdk_heap_free(ctx_p);
    
}

abcdk_spinlock_t *abcdk_spinlock_create()
{
    abcdk_spinlock_t *ctx;

    ctx = (abcdk_spinlock_t *)abcdk_heap_alloc(sizeof(abcdk_spinlock_t));
    if(!ctx)
        return NULL;

    pthread_spin_init(&ctx->spinlock, PTHREAD_PROCESS_PRIVATE);

    return ctx;
}   

int abcdk_spinlock_lock(abcdk_spinlock_t *ctx, int block)
{
    int err = -1;

    assert(ctx);

    if(block)
        err = pthread_spin_lock(&ctx->spinlock);
    else 
        err = pthread_spin_trylock(&ctx->spinlock);

    return err;
}

int abcdk_spinlock_unlock(abcdk_spinlock_t* ctx)
{
    int err = -1;

    assert(ctx);

    err = pthread_spin_unlock(&ctx->spinlock);
    
    return err;
}
