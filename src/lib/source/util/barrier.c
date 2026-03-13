/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2026 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/barrier.h"

/**
 * 屏障.
*/
struct _abcdk_barrier
{
    pthread_barrier_t barrier;
    pthread_barrierattr_t barrierattr;
} ;//abcdk_barrier_t;


void abcdk_barrier_destroy(abcdk_barrier_t **ctx)
{
    abcdk_barrier_t *ctx_p = NULL;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    pthread_barrierattr_destroy(&ctx_p->barrierattr);
    pthread_barrier_destroy(&ctx_p->barrier);
    abcdk_heap_free(ctx_p);
}

abcdk_barrier_t *abcdk_barrier_create(size_t count)
{
    abcdk_barrier_t *ctx;

    assert(count > 0);

    ctx = (abcdk_barrier_t *)abcdk_heap_alloc(sizeof(abcdk_barrier_t));
    if (!ctx)
        return NULL;

    pthread_barrierattr_init(&ctx->barrierattr);
    pthread_barrierattr_setpshared(&ctx->barrierattr, PTHREAD_PROCESS_PRIVATE);

    pthread_barrier_init(&ctx->barrier, &ctx->barrierattr, count);

    return ctx;
}

int abcdk_barrier_wait(abcdk_barrier_t *ctx)
{
    assert(ctx != NULL);

    return pthread_barrier_wait(&ctx->barrier);
}

