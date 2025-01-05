/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/mutex.h"

/**
 * 互斥量、事件。
*/
struct _abcdk_mutex
{
    /**
     * 事件属性。
    */
    pthread_condattr_t condattr;

    /**
     * 事件。
    */
    pthread_cond_t cond;

    /**
     * 互斥量属性。
    */
    pthread_mutexattr_t mutexattr;

    /**
     * 互斥量。
    */
    pthread_mutex_t mutex;

} ;//abcdk_mutex_t;

void abcdk_mutex_destroy(abcdk_mutex_t **ctx)
{
    abcdk_mutex_t *ctx_p = NULL;


    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    pthread_condattr_destroy(&ctx_p->condattr);
    pthread_cond_destroy(&ctx_p->cond);
    pthread_mutexattr_destroy(&ctx_p->mutexattr);
    pthread_mutex_destroy(&ctx_p->mutex);
    abcdk_heap_free(ctx_p);
    
}

abcdk_mutex_t *abcdk_mutex_create()
{
    abcdk_mutex_t *ctx;

    ctx = (abcdk_mutex_t *)abcdk_heap_alloc(sizeof(abcdk_mutex_t));
    if(!ctx)
        return NULL;

    pthread_condattr_init(&ctx->condattr);
    pthread_condattr_setclock(&ctx->condattr, CLOCK_MONOTONIC);
    pthread_condattr_setpshared(&ctx->condattr,PTHREAD_PROCESS_PRIVATE);

    pthread_mutexattr_init(&ctx->mutexattr);
#if !defined(__ANDROID__)
    pthread_mutexattr_setpshared(&ctx->mutexattr,PTHREAD_PROCESS_PRIVATE);
    pthread_mutexattr_setrobust(&ctx->mutexattr,PTHREAD_MUTEX_ROBUST);
#ifdef __USE_GNU
    pthread_mutexattr_settype(&ctx->mutexattr, PTHREAD_MUTEX_FAST_NP);
#endif //__USE_GNU
#endif //__ANDROID__

    pthread_cond_init(&ctx->cond, &ctx->condattr);
    pthread_mutex_init(&ctx->mutex, &ctx->mutexattr);

    return ctx;
}   

int abcdk_mutex_lock(abcdk_mutex_t *ctx, int block)
{
    int err = -1;

    assert(ctx);

    if(block)
        err = pthread_mutex_lock(&ctx->mutex);
    else 
        err = pthread_mutex_trylock(&ctx->mutex);

    /*当互斥量的拥有者异外结束时，恢复互斥量状态的一致性。*/
    if (err == EOWNERDEAD)
    {
#if !defined(__ANDROID__)
        pthread_mutex_consistent(&ctx->mutex);
#endif //__ANDROID__
        pthread_mutex_unlock(&ctx->mutex);
        /*回调自己，重试。*/
        err = abcdk_mutex_lock(ctx,block);
    }    

    return err;
}

int abcdk_mutex_unlock(abcdk_mutex_t* ctx)
{
    int err = -1;

    assert(ctx);

    err = pthread_mutex_unlock(&ctx->mutex);
    
    return err;
}

int abcdk_mutex_wait(abcdk_mutex_t* ctx,time_t timeout)
{
    int err = -1;
    struct timespec sys_ts;
    struct timespec out_ts;
    __clockid_t condclock;

    assert(ctx);

    if (timeout >= 0)
    {
        err = pthread_condattr_getclock(&ctx->condattr, &condclock);
        if (err != 0)
            return err;

        if (condclock == CLOCK_MONOTONIC)
            clock_gettime(CLOCK_MONOTONIC, &sys_ts);
        else if (condclock == CLOCK_REALTIME)
            clock_gettime(CLOCK_REALTIME, &sys_ts);
        else
            ABCDK_ERRNO_AND_RETURN1(EINVAL, err = -1);

        out_ts.tv_sec = sys_ts.tv_sec + (timeout / 1000);
        out_ts.tv_nsec = sys_ts.tv_nsec + (timeout % 1000) * 1000000;

        /*纳秒时间必须小于1秒，因此可能存在进位。*/
        out_ts.tv_sec += out_ts.tv_nsec / 1000000000L;
        out_ts.tv_nsec = out_ts.tv_nsec % 1000000000L;

        err = pthread_cond_timedwait(&ctx->cond, &ctx->mutex, &out_ts);
    }
    else
    {
        err = pthread_cond_wait(&ctx->cond, &ctx->mutex);
    }

    return err;
}

int abcdk_mutex_signal(abcdk_mutex_t* ctx,int broadcast)
{
    int err = -1;

    assert(ctx);

    if(broadcast)
        err = pthread_cond_broadcast(&ctx->cond);
    else
        err = pthread_cond_signal(&ctx->cond);
    
    return err;
}

