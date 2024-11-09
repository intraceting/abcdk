/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/queue.h"

/** 简单的队列。*/
struct _abcdk_queue
{
    /**同步锁。*/
    abcdk_mutex_t *locker;

    /**队列。*/
    abcdk_tree_t *qlist;

    /**计数器。*/
    uint64_t count;

    /** 消息销毁回调函数。*/
    abcdk_queue_msg_destroy_cb msg_destroy_cb;

};// abcdk_queue_t;

void abcdk_queue_free(abcdk_queue_t **ctx)
{
    abcdk_queue_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_tree_free(&ctx_p->qlist);
    abcdk_mutex_destroy(&ctx_p->locker);
    abcdk_heap_free(ctx_p);
}

abcdk_queue_t *abcdk_queue_alloc(abcdk_queue_msg_destroy_cb cb)
{
    abcdk_queue_t *ctx;

    ctx = abcdk_heap_alloc(sizeof(abcdk_queue_t));
    if (!ctx)
        return NULL;

    ctx->locker = abcdk_mutex_create();
    ctx->msg_destroy_cb = cb;

    ctx->qlist = abcdk_tree_alloc3(1);
    if (!ctx->qlist)
        goto final_error;

    return ctx;

final_error:

    abcdk_queue_free(&ctx);

    return NULL;
}

uint64_t abcdk_queue_length(abcdk_queue_t *ctx)
{
    assert(ctx != NULL);

    return ctx->count;
}

void abcdk_queue_unlock(abcdk_queue_t *ctx)
{
    assert(ctx != NULL);

    abcdk_mutex_unlock(ctx->locker);
}

void abcdk_queue_lock(abcdk_queue_t *ctx)
{
    assert(ctx != NULL);

    abcdk_mutex_lock(ctx->locker, 1);
}

void abcdk_queue_signal(abcdk_queue_t *ctx,int broadcast)
{
    assert(ctx != NULL);

    abcdk_mutex_signal(ctx->locker,broadcast);
}

int abcdk_queue_wait(abcdk_queue_t *ctx, time_t timeout)
{
    assert(ctx != NULL);

    return abcdk_mutex_wait(ctx->locker, timeout);
}

void _abcdk_queue_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_queue_t *queue_p = NULL;
    void *msg_p = NULL;

    queue_p = (abcdk_queue_t *)opaque;

    /*复制数据，解除绑定关系。*/
    msg_p = (void*)alloc->pptrs[0];
    alloc->pptrs[0] = NULL;

    if(!msg_p)
        return;

    ABCDK_ASSERT(queue_p->msg_destroy_cb,"未注册销毁函数，消息对象无法销毁。");
    queue_p->msg_destroy_cb(msg_p);
}

int abcdk_queue_push(abcdk_queue_t *ctx, void *msg)
{
    abcdk_tree_t *msg_node;

    assert(ctx != NULL && msg != NULL);

    msg_node = abcdk_tree_alloc3(1);
    if (!msg_node)
        return -1;

    /*注册消息对象释放函数。*/
    abcdk_object_atfree(msg_node->obj, _abcdk_queue_destroy_cb, ctx);

    /*绑定到节点，添加到队列末尾。*/
    msg_node->obj->pptrs[0] = (uint8_t *)msg;
    abcdk_tree_insert2(ctx->qlist, msg_node, 0);

    /*+1.*/
    ctx->count += 1;

    return 0;
}

void *abcdk_queue_pop(abcdk_queue_t *ctx)
{
    abcdk_tree_t *msg_node = NULL;
    void *msg_p = NULL;
    int chk;

    assert(ctx != NULL);

    msg_node = abcdk_tree_child(ctx->qlist, 1);
    if (!msg_node)
        return NULL;

    /*断开节点。*/
    abcdk_tree_unlink(msg_node);

    /*复制消息对象指针，解除绑定关系。*/
    msg_p = (void *)msg_node->obj->pptrs[0];
    msg_node->obj->pptrs[0] = NULL;

    /*删除节点。*/
    abcdk_tree_free(&msg_node);

    /*-1.*/
    ctx->count -= 1;

    return msg_p;
}
