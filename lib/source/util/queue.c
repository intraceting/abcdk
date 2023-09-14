/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/queue.h"

/** 队列。*/
struct _abcdk_queue
{
    /** 锁。*/
    abcdk_mutex_t locker;

    /** 队列。*/
    abcdk_tree_t *root;

    /** 长度。*/
    size_t count;

    /** 消息销毁回调函数。*/
    abcdk_queue_msg_destroy_cb msg_destroy_cb;

};// abcdk_queue_t;

void abcdk_queue_free(abcdk_queue_t **queue)
{
    abcdk_queue_t *queue_p;

    if (!queue || !*queue)
        return;

    queue_p = *queue;
    *queue = NULL;

    abcdk_tree_free(&queue_p->root);
    abcdk_mutex_destroy(&queue_p->locker);
    abcdk_heap_free(queue_p);
}

abcdk_queue_t *abcdk_queue_alloc(abcdk_queue_msg_destroy_cb cb)
{
    abcdk_queue_t *queue;

    queue = abcdk_heap_alloc(sizeof(abcdk_queue_t));
    if (!queue)
        return NULL;

    queue->count = 0;
    abcdk_mutex_init2(&queue->locker, 0);
    queue->msg_destroy_cb = cb;

    queue->root = abcdk_tree_alloc3(1);
    if (!queue->root)
        goto final_error;

    return queue;

final_error:

    abcdk_queue_free(&queue);

    return NULL;
}

size_t abcdk_queue_count(abcdk_queue_t *queue)
{
    size_t count;

    assert(queue != NULL);

    abcdk_mutex_lock(&queue->locker, 1);
    count = queue->count;
    abcdk_mutex_unlock(&queue->locker);

    return count;
}

void _abcdk_queue_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_queue_t *queue_p = NULL;

    queue_p = (abcdk_queue_t *)opaque;

    /*可能已经解除绑定关系了。*/
    if(!alloc->pptrs[0])
        return;

    if(queue_p->msg_destroy_cb)
        queue_p->msg_destroy_cb((void *)alloc->pptrs[0]);
}

int abcdk_queue_push(abcdk_queue_t *queue, size_t refcount, const void *msg, int head)
{
    abcdk_tree_t *msg_node;

    assert(queue != NULL && refcount > 0 && msg != NULL);

    size_t sizes[] = {0,sizeof(size_t)};
    msg_node = abcdk_tree_alloc2(sizes,2,0);
    if (!msg_node)
        return -1;

    /*注册消息对象释放函数。*/
    abcdk_object_atfree(msg_node->obj, _abcdk_queue_destroy_cb, queue);

    /*绑定到节点。*/
    msg_node->obj->pptrs[0] = (uint8_t *)msg;
    ABCDK_PTR2SIZE(msg_node->obj->pptrs[1],0) = refcount;

    abcdk_mutex_lock(&queue->locker, 1);

    abcdk_tree_insert2(queue->root, msg_node, head);
    queue->count += 1;

    abcdk_mutex_signal(&queue->locker,1);

    abcdk_mutex_unlock(&queue->locker);

    return 0;
}

const void *abcdk_queue_pop(abcdk_queue_t *queue, int head)
{
    abcdk_tree_t *msg_node = NULL;
    const void *msg_p = NULL;

    assert(queue != NULL);

    abcdk_mutex_lock(&queue->locker, 1);

    msg_node = abcdk_tree_child(queue->root, head);
    if (msg_node)
    {
        /*复制消息对象指针*/
        msg_p = (void *)msg_node->obj->pptrs[0];

        /*最后一个引用者取走数据后，则能删除节点。*/
        if (ABCDK_PTR2SIZE(msg_node->obj->pptrs[1], 0) == 1)
        {
            abcdk_tree_unlink(msg_node);
            queue->count -= 1;
        }
        else
        {
            /*引用计数减1。*/
            ABCDK_PTR2SIZE(msg_node->obj->pptrs[1], 0) -= 1;
            msg_node = NULL; // 未达到删除条件。
        }
    }

    abcdk_mutex_unlock(&queue->locker);

    if (msg_node)
    {
        /*解除绑定关系。*/
        msg_node->obj->pptrs[0] = NULL;
        abcdk_tree_free(&msg_node);
    }

    return msg_p;
}

int abcdk_queue_wait(abcdk_queue_t *queue, time_t timeout)
{
    int chk;

    assert(queue != NULL && timeout > 0);

    abcdk_mutex_lock(&queue->locker, 1);

    if (queue->count <= 0)
        chk = abcdk_mutex_wait(&queue->locker, timeout);
    else 
        chk = 0;

    abcdk_mutex_unlock(&queue->locker);

    return chk;
}
