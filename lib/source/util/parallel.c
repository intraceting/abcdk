/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/parallel.h"

/** 并行计算环境。*/
struct _abcdk_parallel
{
    /**
     * 指令。
     *
     * 1：停止，2：运行。
     */
    volatile int worker_cmd;

    /**线程数组。*/
    abcdk_thread_t *work_threads;

    /**工作队列。*/
    abcdk_queue_t *work_queue;

    /**线程数量。*/
    int thread_count;


}; // abcdk_parallel_t;

/** 并行计算项目。*/
typedef struct _abcdk_parallel_item
{
    /**同步器。*/
    abcdk_mutex_t sync;

    /**执行计数器。*/
    uint32_t counter;

    /**线程编号索引。*/
    volatile uint32_t idx;

    /** 线程数量。*/
    uint32_t number;

    /**项目环境指针。*/
    void *opaque;

    /**回调函数。*/
    abcdk_parallel_routine_cb routine_cb;

} abcdk_parallel_item_t;

void _abcdk_parallel_item_free(abcdk_parallel_item_t **item)
{
    abcdk_parallel_item_t *item_p;

    if (!item || !*item)
        return;

    item_p = *item;
    *item = NULL;

    abcdk_mutex_destroy(&item_p->sync);

    abcdk_heap_free(item_p);
}

abcdk_parallel_item_t *_abcdk_parallel_item_alloc()
{
    abcdk_parallel_item_t *item;

    item = (abcdk_parallel_item_t *)abcdk_heap_alloc(sizeof(abcdk_parallel_item_t));
    if (!item)
        return NULL;

    abcdk_mutex_init2(&item->sync, 0);

    return item;
}


void _abcdk_parallel_free(abcdk_parallel_t *ctx)
{

    abcdk_heap_free(ctx->work_threads);
    abcdk_queue_free(&ctx->work_queue);

    abcdk_heap_free(ctx);
}

abcdk_parallel_t *_abcdk_parallel_alloc(size_t max)
{
    abcdk_parallel_t *ctx;

    ctx = (abcdk_parallel_t*)abcdk_heap_alloc(sizeof(abcdk_parallel_t));
    if(!ctx)
        return NULL;

    ctx->worker_cmd = 1;
    ctx->thread_count = max;

    ctx->work_threads = abcdk_heap_alloc(sizeof(abcdk_thread_t) * max);
    if(!ctx->work_threads)
        goto final_error;

    ctx->work_queue = abcdk_queue_alloc(NULL);
    if(!ctx->work_queue)
        goto final_error;

    

    return ctx;

final_error:

    return NULL;
}

void *_abcdk_parallel_routine(void *opaque)
{
    abcdk_parallel_t *ctx = (abcdk_parallel_t *)opaque;
    abcdk_parallel_item_t *item;
    uint32_t tid;

wait_next_item:

    item = (abcdk_parallel_item_t *)abcdk_queue_pop(ctx->work_queue, 1);
    if (!item)
    {
        if (!abcdk_atomic_compare(&ctx->worker_cmd, 2))
            goto final;

        abcdk_queue_wait(ctx->work_queue, 1000);
        goto wait_next_item;
    }

    tid = abcdk_atomic_fetch_and_add(&item->idx, 1);

    item->routine_cb(item->opaque, tid);

    abcdk_mutex_lock(&item->sync,1);
        
    /*仅需要通知一次。*/
    if(++item->counter == item->number)
        abcdk_mutex_signal(&item->sync,1);

    abcdk_mutex_unlock(&item->sync);
    
    goto wait_next_item;

final:

    return NULL;
}

void _abcdk_parallel_stop(abcdk_parallel_t *ctx)
{
    /*未启动，则不需要停止；已启动，则停止一次限可。*/
    if (!abcdk_atomic_compare_and_swap(&ctx->worker_cmd, 2, 1))
        return;

    /*等待所有计算完成。*/
    for (int i = 0; i < ctx->thread_count; i++)
        abcdk_thread_join(&ctx->work_threads[i]);

    /*走到这里表示所有的都计算完成了。*/
    assert(abcdk_queue_count(ctx->work_queue) == 0);
}

int _abcdk_parallel_start(abcdk_parallel_t *ctx)
{
    static volatile uint64_t cpu_bind_idx = 0;
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    int chk;
    
    /*启动一次即可。*/
    if (abcdk_atomic_compare(&ctx->worker_cmd, 2))
        return 0;

    /*设置启动命令。*/
    abcdk_atomic_store(&ctx->worker_cmd, 2);

    for (int i = 0; i < ctx->thread_count; i++)
    {
        ctx->work_threads[i].opaque = ctx;
        ctx->work_threads[i].routine = _abcdk_parallel_routine;

        chk = abcdk_thread_create(&ctx->work_threads[i], 1);
        if (chk != 0)
            goto final_error;

        /*同一个进程中的多个对象，CPU尽可能匀衡分布。*/
        uint64_t cpu_idx = abcdk_atomic_fetch_and_add(&cpu_bind_idx,1);

        /*设置亲源CPU，加速运行。*/
        abcdk_thread_setaffinity2(ctx->work_threads[i].handle,cpu_idx%cpus);
    }

    return 0;

final_error:

    /*启动失败，恢复停止命令。*/
    abcdk_atomic_store(&ctx->worker_cmd, 1);
    return -1;
}

void abcdk_parallel_free(abcdk_parallel_t **ctx)
{
    abcdk_parallel_t *ctx_p;
    abcdk_parallel_item_t *item;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    _abcdk_parallel_stop(ctx_p);
    _abcdk_parallel_free(ctx_p);
}

abcdk_parallel_t *abcdk_parallel_alloc(size_t max)
{
    abcdk_parallel_t *ctx;
    int chk;

    assert(max > 0);

    ctx = _abcdk_parallel_alloc(max);
    if(!ctx)
        return NULL;

    chk = _abcdk_parallel_start(ctx);
    if(chk != 0)
        goto final_error;

    return ctx;

final_error:

    abcdk_parallel_free(&ctx);

    return NULL;
}

int abcdk_parallel_run(abcdk_parallel_t *ctx,uint32_t number, void *opaque, abcdk_parallel_routine_cb routine_cb)
{
    abcdk_parallel_item_t *item;
    int chk;

    assert(ctx != NULL && number > 0 && routine_cb != NULL);

    item = _abcdk_parallel_item_alloc();
    if (!item)
        return -1;

    item->idx = 0;
    item->counter = 0;
    item->number = number;
    item->opaque = opaque;
    item->routine_cb = routine_cb;

    chk = abcdk_queue_push(ctx->work_queue, number, item, 0);
    if (chk != 0)
        goto final;


    abcdk_mutex_lock(&item->sync,1);

    for(;;)
    {
        if(item->counter != item->number)
            abcdk_mutex_wait(&item->sync,-1);
        else 
            break;
    }

    abcdk_mutex_unlock(&item->sync);

    chk = 0;

final:

    _abcdk_parallel_item_free(&item);

    return chk;
}