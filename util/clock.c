/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/clock.h"

/**
 * 计时器
*/
typedef struct _abcdk_clock
{
    /**
     * 起始
    */
    uint64_t start;

    /**
     * 打点
    */
    uint64_t point;

} abcdk_clock_t;

void _abcdk_clock_destroy(void* buf)
{
    if(buf)
        abcdk_heap_free(buf);
}

int _abcdk_clock_create(void *opaque)
{
    return pthread_key_create((pthread_key_t *)opaque, _abcdk_clock_destroy);
}

abcdk_clock_t *_abcdk_clock_init(uint64_t set)
{
    static volatile int init = 0;
    static pthread_key_t key = -1;

    int chk = 0;
    abcdk_clock_t *ctx = NULL;

    chk = abcdk_once(&init, _abcdk_clock_create, &key);
    if (chk < 0)
        return NULL;

    ctx = pthread_getspecific(key);
    if (!ctx)
    {
        ctx = abcdk_heap_alloc(sizeof(abcdk_clock_t));
        if (!ctx)
            return NULL;

        chk = pthread_setspecific(key, ctx);
        if (chk != 0)
            abcdk_heap_free2((void **)&ctx);
    }

    if (ctx && chk == 0)
        ctx->point = ctx->start = set;

    return ctx;
}

void abcdk_clock_reset()
{
    uint64_t current = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,6);
    abcdk_clock_t* ctx = _abcdk_clock_init(current);

    assert(ctx);
    
    ctx->start = ctx->point = current;
}

uint64_t abcdk_clock_dot(uint64_t *step)
{
    uint64_t current = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,6);
    abcdk_clock_t* ctx = _abcdk_clock_init(current);

    assert(ctx);

    uint64_t dot = current - ctx->start;

    if (step)
        *step = current - ctx->point;

    ctx->point = current;

    return dot;
}

uint64_t abcdk_clock_step(uint64_t *dot)
{
    uint64_t ldot = 0, step = 0;
    
    ldot = abcdk_clock_dot(&step);

    if(dot)
        *dot = ldot;
    
    return step;
}