/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "comm/queue.h"

/**/
//#define ABCDK_COMM_QUEUE_SPINLOCK   1

/** 消息队列。*/
typedef struct _abcdk_comm_queue
{
    /** 锁。*/
#ifndef ABCDK_COMM_QUEUE_SPINLOCK 
    abcdk_mutex_t locker;
#else 
    pthread_spinlock_t locker;
#endif 

    /** 队列。*/
    abcdk_tree_t *root;

    /** 长度。*/
    size_t count;

} abcdk_comm_queue_t;

void abcdk_comm_queue_free(abcdk_comm_queue_t **queue)
{
    abcdk_comm_queue_t *queue_p;

    if (!queue || !*queue)
        return;

    queue_p = *queue;

    abcdk_tree_free(&queue_p->root);
#ifndef ABCDK_COMM_QUEUE_SPINLOCK 
    abcdk_mutex_destroy(&queue_p->locker);
#else 
    pthread_spin_destroy(&queue_p->locker);
#endif 

    abcdk_heap_free(queue_p);

    /*Set NULL(0).*/
    *queue = NULL;
}

abcdk_comm_queue_t *abcdk_comm_queue_alloc()
{
    abcdk_comm_queue_t *queue;

    queue = abcdk_heap_alloc(sizeof(abcdk_comm_queue_t));
    if (!queue)
        return NULL;

    queue->count = 0;
#ifndef ABCDK_COMM_QUEUE_SPINLOCK 
    abcdk_mutex_init2(&queue->locker, 0);
#else 
    pthread_spin_init(&queue->locker,PTHREAD_PROCESS_PRIVATE);
#endif 
    queue->root = abcdk_tree_alloc3(1);
    if (!queue->root)
        goto final_error;

    return queue;

final_error:

    abcdk_comm_queue_free(&queue);

    return NULL;
}

size_t abcdk_comm_queue_count(abcdk_comm_queue_t *queue)
{
    assert(queue != NULL);

    return queue->count;
}

void _abcdk_comm_queue_destroy_cb(abcdk_allocator_t *alloc, void *opaque)
{
    abcdk_comm_message_t *msg_p = NULL;

    msg_p = (abcdk_comm_message_t *)alloc->pptrs[0];

    abcdk_comm_message_unref(&msg_p);
}

int abcdk_comm_queue_push(abcdk_comm_queue_t *queue, abcdk_comm_message_t *msg)
{
    abcdk_tree_t *msg_node;

    assert(queue != NULL && msg != NULL);

    msg_node = abcdk_tree_alloc3(0);
    if (!msg_node)
        return -1;

    /*注册消息对象释放函数。*/
    abcdk_allocator_atfree(msg_node->alloc, _abcdk_comm_queue_destroy_cb, NULL);

    /*绑定到节点。*/
    msg_node->alloc->pptrs[0] = (uint8_t *)msg;

#ifndef ABCDK_COMM_QUEUE_SPINLOCK 
    abcdk_mutex_lock(&queue->locker, 1);
#else 
    pthread_spin_lock(&queue->locker);
#endif 

    abcdk_tree_insert2(queue->root, msg_node, 0);
    queue->count += 1;

#ifndef ABCDK_COMM_QUEUE_SPINLOCK 
    abcdk_mutex_unlock(&queue->locker);
#else 
    pthread_spin_unlock(&queue->locker);
#endif 

    return 0;
}

abcdk_comm_message_t *abcdk_comm_queue_pop(abcdk_comm_queue_t *queue)
{
    abcdk_tree_t *msg_node = NULL;
    abcdk_comm_message_t *msg_p = NULL;

    assert(queue != NULL);

#ifndef ABCDK_COMM_QUEUE_SPINLOCK 
    abcdk_mutex_lock(&queue->locker, 1);
#else 
    pthread_spin_lock(&queue->locker);
#endif 

    msg_node = abcdk_tree_child(queue->root, 1);
    if (msg_node)
    {
        abcdk_tree_unlink(msg_node);
        queue->count -= 1;
    }

#ifndef ABCDK_COMM_QUEUE_SPINLOCK 
    abcdk_mutex_unlock(&queue->locker);
#else 
    pthread_spin_unlock(&queue->locker);
#endif 

    if (!msg_node)
        return NULL;

    /*复制消息对象指针，解除绑定关系。*/
    msg_p = (abcdk_comm_message_t *)msg_node->alloc->pptrs[0];
    msg_node->alloc->pptrs[0] = NULL;

    abcdk_tree_free(&msg_node);

    return msg_p;
}

