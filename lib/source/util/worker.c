/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/worker.h"

/**简单的线程池。*/
struct _abcdk_worker
{
    /**配置。*/
    abcdk_worker_config_t cfg;

    /**退出标志。0：运行，1：停止。*/
    int exit_flag;

    /**线程数组。*/
    abcdk_thread_t *threads_ctx;

    /**项目队列。*/
    abcdk_queue_t *queue_ctx;


}; // abcdk_worker_t;

/** 工作项目。*/
typedef struct _abcdk_worker_item
{
    /*事件。*/
    uint64_t event;

    /*数据。*/
    void *data;

} abcdk_worker_item_t;

void _abcdk_worker_item_free(abcdk_worker_item_t **item)
{
    abcdk_worker_item_t *item_p;

    if (!item || !*item)
        return;

    item_p = *item;
    *item = NULL;

    abcdk_heap_free(item_p);
}

abcdk_worker_item_t *_abcdk_worker_item_alloc()
{
    abcdk_worker_item_t *item;

    item = (abcdk_worker_item_t *)abcdk_heap_alloc(sizeof(abcdk_worker_item_t));
    if (!item)
        return NULL;

    item->event = 0;
    item->data = NULL;

    return item;
}

void *_abcdk_worker_routine(void *opaque)
{
    abcdk_worker_t *ctx = (abcdk_worker_t *)opaque;
    abcdk_worker_item_t *item_p;

NEXT:

    /*清理野指针。*/
    item_p = NULL;

    abcdk_queue_lock(ctx->queue_ctx);

    while(1)
    {
        item_p = (abcdk_worker_item_t *)abcdk_queue_pop(ctx->queue_ctx);
        if (item_p)
            break;

        /*检查退出标志。*/
        if(ctx->exit_flag != 0)
            break;
        else 
            abcdk_queue_wait(ctx->queue_ctx,-1);
    }

    abcdk_queue_unlock(ctx->queue_ctx);

    /*只能通知退出时才会生效。*/
    if(!item_p)
        return NULL;

    ctx->cfg.process_cb(ctx->cfg.opaque, item_p->event,item_p->data);
    _abcdk_worker_item_free(&item_p);
    
    goto NEXT;
}

int _abcdk_worker_start(abcdk_worker_t *ctx)
{
    static volatile int cpu_idx = 0;
    int cpu_set;
    int chk;

    for (int i = 0; i < ctx->cfg.numbers; i++)
    {
        ctx->threads_ctx[i].opaque = ctx;
        ctx->threads_ctx[i].routine = _abcdk_worker_routine;

        chk = abcdk_thread_create(&ctx->threads_ctx[i], 1);
        if (chk != 0)
            return -1;

        /*尽可能让线程分布在不同的核心上。*/
        cpu_set = abcdk_atomic_add_and_fetch(&cpu_idx, 1) % sysconf(_SC_NPROCESSORS_ONLN);
        
#if !defined(__ANDROID__)
        abcdk_thread_setaffinity2(ctx->threads_ctx[i].handle,cpu_set);
#endif //__ANDROID__
    }

    return 0;
}

void abcdk_worker_stop(abcdk_worker_t **ctx)
{
    abcdk_worker_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;

    /*通知退出。*/
    abcdk_queue_lock(ctx_p->queue_ctx);
    ctx_p->exit_flag = 1;
    abcdk_queue_signal(ctx_p->queue_ctx,1);
    abcdk_queue_unlock(ctx_p->queue_ctx);

    /*等待所有线程结束，并删除线程数组。*/
    if(ctx_p->threads_ctx)
    {
        for (int i = 0; i < ctx_p->cfg.numbers; i++)
            abcdk_thread_join(&ctx_p->threads_ctx[i]);

        abcdk_heap_free(ctx_p->threads_ctx);
    }

    /*删除队列。*/
    abcdk_queue_free(&ctx_p->queue_ctx);

    abcdk_heap_free(ctx_p);

    /*一定要等WORKER对象停下来才能清空指针，否则会因为线程调度问题造成引用空指针。*/
    *ctx = NULL;
}

abcdk_worker_t *abcdk_worker_start(abcdk_worker_config_t *cfg)
{
    abcdk_worker_t *ctx;
    int chk;

    assert(cfg != NULL);
    assert(cfg->process_cb != NULL);

    ctx = (abcdk_worker_t*)abcdk_heap_alloc(sizeof(abcdk_worker_t));
    if(!ctx)
        return NULL;

    ctx->cfg = *cfg;

    if(ctx->cfg.numbers <= 0)
        ctx->cfg.numbers = sysconf(_SC_NPROCESSORS_ONLN);

    ctx->threads_ctx = abcdk_heap_alloc(sizeof(abcdk_thread_t) * ctx->cfg.numbers);
    if(!ctx->threads_ctx)
        goto ERR;

    ctx->queue_ctx = abcdk_queue_alloc(NULL);
    if(!ctx->queue_ctx)
        goto ERR;

    chk = _abcdk_worker_start(ctx);
    if(chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_worker_stop(&ctx);
    return NULL;
}

int abcdk_worker_dispatch(abcdk_worker_t *ctx,uint64_t event,void *item)
{
    abcdk_worker_item_t *item_p;
    int chk;

    assert(ctx != NULL);

    item_p = _abcdk_worker_item_alloc();
    if (!item_p)
        return -1;

    item_p->event = event;
    item_p->data = item;

    abcdk_queue_lock(ctx->queue_ctx);

    chk = abcdk_queue_push(ctx->queue_ctx, item_p);
    if (chk != 0)
        _abcdk_worker_item_free(&item_p);

    abcdk_queue_signal(ctx->queue_ctx,0);

    abcdk_queue_unlock(ctx->queue_ctx);

    return (chk == 0?0: -1);
}
