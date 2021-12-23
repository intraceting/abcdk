/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-comm/message.h"

/** 消息对象。*/
typedef struct _abcdk_comm_msg
{
    /** 引用计数器。*/
    volatile int refcount;

    /** 内存块指针。*/
    void *buf;

    /* 读写偏移量。*/
    size_t offset;

    /** 容量。*/
    size_t capacity;

    /** 长度。*/
    uint32_t size;

    /** 数据包协议回调函数指针。*/
    abcdk_comm_msg_protocol_cb protocol_cb;

}abcdk_comm_msg_t;

/** 消息队列。*/
typedef struct _abcdk_comm_msg_queue
{
    /** 锁。*/
    abcdk_mutex_t locker;

    /** 队列。*/
    abcdk_tree_t *root;

}abcdk_comm_msg_queue_t;

void abcdk_comm_msg_unref(abcdk_comm_msg_t **msg)
{
    abcdk_comm_msg_t *msg_p = NULL;

    if(!msg || !*msg)
        return;

    msg_p = *msg;

    if (abcdk_atomic_fetch_and_add(&msg_p->refcount, -1) != 1)
        goto final;

    assert(msg_p->refcount == 0);

    abcdk_heap_free2(&msg_p->buf);

    abcdk_heap_free(msg_p);

final:

    /*set NULL(0).*/
    *msg = NULL;
}

abcdk_comm_msg_t *abcdk_comm_msg_refer(abcdk_comm_msg_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_comm_msg_t *abcdk_comm_msg_alloc(size_t size)
{
    abcdk_comm_msg_t *msg = NULL;

    assert(size > 0);

    msg = abcdk_heap_alloc(sizeof(abcdk_comm_msg_t));
    if (!msg)
        goto final_error;
    
    msg->refcount = 1;
    msg->offset = 0;
    msg->protocol_cb = NULL;
    msg->size = size;
    msg->capacity = ABCDK_MAX(msg->size,1024UL);
    msg->buf = abcdk_heap_alloc(msg->capacity);

    if (!msg->buf)
        goto final_error;

    return msg;

final_error:

    abcdk_comm_msg_unref(&msg);

    return NULL;
}

int abcdk_comm_msg_realloc(abcdk_comm_msg_t *msg,size_t size)
{
    void * new_buf = NULL;

    assert(msg != NULL && size > 0);

    /*新的大小与旧的大小一样时，不需要调整。*/
    if (msg->size == size)
        goto final;

    msg->size = size;

    /*新的容量与旧的容量一样时，不需要调整。*/
    if (msg->capacity == ABCDK_MAX(msg->size,1024UL))
        goto final;

    msg->capacity = ABCDK_MAX(msg->size,1024UL);

    new_buf = abcdk_heap_realloc(msg->buf,msg->capacity);
    if (!new_buf)
        return -1;

    /*绑定新内存。*/
    msg->buf = new_buf;

final:

    /*修正编移量。*/
    if (msg->offset > msg->size)
        msg->offset = msg->size;

    return 0;
}

void abcdk_comm_msg_reset(abcdk_comm_msg_t *msg)
{
    assert(msg != NULL);

    msg->offset = 0;
}

void *abcdk_comm_msg_data(const abcdk_comm_msg_t *msg)
{
    assert(msg != NULL);

    return msg->buf;
}

size_t abcdk_comm_msg_size(const abcdk_comm_msg_t *msg)
{
    assert(msg != NULL);

    return msg->size;
}

size_t abcdk_comm_msg_offset(const abcdk_comm_msg_t *msg)
{
    assert(msg != NULL);
    
    return msg->offset;
}

void abcdk_comm_msg_protocol_set(abcdk_comm_msg_t *msg,abcdk_comm_msg_protocol_cb protocol_cb)
{
    assert(msg != NULL && protocol_cb != NULL);

    msg->protocol_cb = protocol_cb;
}

int abcdk_comm_msg_recv(abcdk_comm_node_t *node, abcdk_comm_msg_t *msg)
{
    uint32_t size;
    ssize_t rsize;
    int chk;

    assert(node != NULL && msg != NULL);

MORE_DATA:

    rsize = abcdk_comm_read(node, ABCDK_PTR2VPTR(msg->buf, msg->offset), msg->size - msg->offset);
    if (rsize <= 0)
        return 0;
    else if (rsize > 0)
        msg->offset += rsize;

    /*检测接收的数据是否完整。*/
    if(msg->protocol_cb)
    {
        chk = msg->protocol_cb(node,msg);
        if(chk == 0)
            goto MORE_DATA;
    }

    return 1;
}

int abcdk_comm_msg_send(abcdk_comm_node_t *node, abcdk_comm_msg_t *msg)
{
    uint32_t size;
    ssize_t wsize;
    int chk;

    assert(node != NULL && msg != NULL);

    if (msg->offset >= msg->size)
        return 1;

    wsize = abcdk_comm_write(node, ABCDK_PTR2VPTR(msg->buf, msg->offset), msg->size - msg->offset);
    if (wsize <= 0)
        return 0;
    else if (wsize > 0)
        msg->offset += wsize;

    /*检测发送的数据是否完整。*/
    if (msg->size != msg->offset)
        return 0;
        
    return 1;
}

void abcdk_comm_msg_queue_free(abcdk_comm_msg_queue_t **queue)
{
    abcdk_comm_msg_queue_t *queue_p;

    if(!queue || !*queue)
        return;

    queue_p = *queue;

    abcdk_tree_free(&queue_p->root);
    abcdk_mutex_destroy(&queue_p->locker);

    abcdk_heap_free(queue_p);

    /*Set NULL(0).*/
    *queue = NULL;
}

abcdk_comm_msg_queue_t *abcdk_comm_msg_queue_alloc()
{
    abcdk_comm_msg_queue_t *queue;

    queue = abcdk_heap_alloc(sizeof(abcdk_comm_msg_queue_t));
    if(!queue)
        return NULL;

    abcdk_mutex_init2(&queue->locker, 0);
    queue->root = abcdk_tree_alloc3(1);
    if (!queue->root)
        goto final_error;

    return queue;

final_error:

    abcdk_comm_msg_queue_free(&queue);

    return NULL;
}

void _abcdk_comm_msg_queue_destroy_cb(abcdk_allocator_t *alloc, void *opaque)
{
    abcdk_comm_msg_t *msg_p = NULL;

    msg_p = (abcdk_comm_msg_t *)alloc->pptrs[0];

    abcdk_comm_msg_unref(&msg_p);
}

int abcdk_comm_msg_queue_push(abcdk_comm_msg_queue_t *queue, abcdk_comm_msg_t *msg)
{
    abcdk_tree_t *msg_node;

    assert(queue != NULL && msg != NULL);

    msg_node = abcdk_tree_alloc3(0);
    if (!msg_node)
        return -1;

    /*注册消息对象释放函数。*/
    abcdk_allocator_atfree(msg_node->alloc, _abcdk_comm_msg_queue_destroy_cb, NULL);

    /*绑定到节点。*/
    msg_node->alloc->pptrs[0] = (uint8_t *)msg;

    abcdk_mutex_lock(&queue->locker, 1);
    abcdk_tree_insert2(queue->root, msg_node, 0);
    abcdk_mutex_unlock(&queue->locker);

    return 0;
}

abcdk_comm_msg_t *abcdk_comm_msg_queue_pop(abcdk_comm_msg_queue_t *queue)
{
    abcdk_tree_t *msg_node = NULL;
    abcdk_comm_msg_t *msg_p = NULL;

    abcdk_mutex_lock(&queue->locker, 1);
    msg_node = abcdk_tree_child(queue->root, 1);
    if (msg_node)
        abcdk_tree_unlink(msg_node);
    abcdk_mutex_unlock(&queue->locker);

    if(!msg_node)
        return NULL;

    /*复制消息对象指针，解除绑定关系。*/
    msg_p = (abcdk_comm_msg_t*)msg_node->alloc->pptrs[0];
    msg_node->alloc->pptrs[0] = NULL;

    abcdk_tree_free(&msg_node);

    return msg_p;
}
