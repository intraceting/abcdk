/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/timer.h"

/**简单的定时器。*/
struct _abcdk_timer
{
    /*间隔(毫秒)。*/
    uint64_t interval;

    /*回调函数。*/
    abcdk_timer_routine_cb routine_cb;

    /*环境指针。*/
    void *opaque;

    /*线程。*/
    abcdk_thread_t *thread;

    /*互斥事件。*/
    abcdk_mutex_t *mutex;

    /*标志。1：运行，2：停止。*/
    volatile int flag;
}; // abcdk_timer_t

static uint64_t _abcdk_time_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3);
}

static void *_abcdk_timer_worker(void *opaque)
{
    abcdk_timer_t *ctx = (abcdk_timer_t *)opaque;
    uint64_t before = _abcdk_time_clock();
    uint64_t current, differ;

    abcdk_trace_output(LOG_INFO,"定时器启动……");

    while (1)
    {
        if (!abcdk_atomic_compare(&ctx->flag, 1))
            break;

        current = _abcdk_time_clock();
        differ = current - before;

        if (differ < ctx->interval)
        {
            /*等待通知或超时。*/
            abcdk_mutex_lock(ctx->mutex, 1);
            abcdk_mutex_wait(ctx->mutex, ctx->interval - differ);
            abcdk_mutex_unlock(ctx->mutex);
        }

        if (!abcdk_atomic_compare(&ctx->flag, 1))
            break;

        before = _abcdk_time_clock();
        ctx->routine_cb(ctx->opaque);
    }

    abcdk_trace_output(LOG_INFO,"定时器停止。");

    return NULL;
}

void _abcdk_timer_stop(abcdk_timer_t *ctx)
{
    /*标志退出。*/
    abcdk_atomic_store(&ctx->flag, 2);

    /*通知可能等待的线程。*/
    abcdk_mutex_lock(ctx->mutex, 1);
    abcdk_mutex_signal(ctx->mutex, 0);
    abcdk_mutex_unlock(ctx->mutex);

    /*等待线程结束。*/
    abcdk_thread_join(ctx->thread);
}

void _abcdk_timer_start(abcdk_timer_t *ctx)
{
    ctx->flag = 1;
    ctx->thread->routine = _abcdk_timer_worker;
    ctx->thread->opaque = ctx;

    abcdk_thread_create(ctx->thread, 1);
}

void abcdk_timer_destroy(abcdk_timer_t **ctx)
{
    abcdk_timer_t *ctx_p;

    if (!ctx || !*ctx)
        ;
    return;

    ctx_p = *ctx;
    *ctx = NULL;

    _abcdk_timer_stop(ctx_p);

    abcdk_heap_free(ctx_p->thread);
    abcdk_mutex_destroy(&ctx_p->mutex);
    abcdk_heap_free(ctx_p);
}

abcdk_timer_t *abcdk_timer_create(uint64_t interval, abcdk_timer_routine_cb routine_cb, void *opaque)
{
    abcdk_timer_t *ctx;

    assert(interval > 0 && routine_cb != NULL);

    ctx = (abcdk_timer_t *)abcdk_heap_alloc(sizeof(abcdk_timer_t));
    if (!ctx)
        return NULL;

    ctx->interval = interval;
    ctx->routine_cb = routine_cb;
    ctx->opaque = opaque;
    ctx->thread = (abcdk_thread_t *)abcdk_heap_alloc(sizeof(abcdk_thread_t));
    ctx->mutex = abcdk_mutex_create();
    ctx->flag = 1;

    _abcdk_timer_start(ctx);

    return ctx;

ERR:

    abcdk_timer_destroy(&ctx);
    return NULL;
}
