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
    /*索引。*/
    uint64_t index;

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

static void *_abcdk_timer_worker(void *opaque)
{
    abcdk_timer_t *ctx = (abcdk_timer_t *)opaque;
    uint64_t interval = 0;

    /*设置线程名字，日志记录会用到。*/
    abcdk_thread_setname(0, "%x", ctx->index);

    abcdk_trace_printf(LOG_INFO, "定时器启动……");

    while (1)
    {
        if (!abcdk_atomic_compare(&ctx->flag, 1))
            break;

        if (interval > 0)
        {
            /*等待通知或超时。*/
            abcdk_mutex_lock(ctx->mutex, 1);
            abcdk_mutex_wait(ctx->mutex, interval);
            abcdk_mutex_unlock(ctx->mutex);
        }

        if (!abcdk_atomic_compare(&ctx->flag, 1))
            break;

        interval = ctx->routine_cb(ctx->opaque);
    }

    abcdk_trace_printf(LOG_INFO, "定时器停止。");

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
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    _abcdk_timer_stop(ctx_p);

    abcdk_heap_free(ctx_p->thread);
    abcdk_mutex_destroy(&ctx_p->mutex);
    abcdk_heap_free(ctx_p);
}

abcdk_timer_t *abcdk_timer_create(abcdk_timer_routine_cb routine_cb, void *opaque)
{
    abcdk_timer_t *ctx;

    assert(routine_cb != NULL);

    ctx = (abcdk_timer_t *)abcdk_heap_alloc(sizeof(abcdk_timer_t));
    if (!ctx)
        return NULL;

    ctx->index = abcdk_sequence_num();
    ctx->routine_cb = routine_cb;
    ctx->opaque = opaque;
    ctx->thread = (abcdk_thread_t *)abcdk_heap_alloc(sizeof(abcdk_thread_t));
    ctx->mutex = abcdk_mutex_create();

    _abcdk_timer_start(ctx);

    return ctx;

ERR:

    abcdk_timer_destroy(&ctx);
    return NULL;
}
