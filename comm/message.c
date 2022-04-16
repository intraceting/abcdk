/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "comm/message.h"

/** 消息对象。*/
typedef struct _abcdk_comm_message
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
    abcdk_comm_message_protocol_cb protocol_cb;

} abcdk_comm_message_t;

void abcdk_comm_message_unref(abcdk_comm_message_t **msg)
{
    abcdk_comm_message_t *msg_p = NULL;

    if (!msg || !*msg)
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

abcdk_comm_message_t *abcdk_comm_message_refer(abcdk_comm_message_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_comm_message_t *abcdk_comm_message_alloc(size_t size)
{
    abcdk_comm_message_t *msg = NULL;

    assert(size > 0);

    msg = abcdk_heap_alloc(sizeof(abcdk_comm_message_t));
    if (!msg)
        goto final_error;

    msg->refcount = 1;
    msg->offset = 0;
    msg->protocol_cb = NULL;
    msg->size = size;
    msg->capacity = ABCDK_MAX(msg->size, 1024UL);
    msg->buf = abcdk_heap_alloc(msg->capacity);

    if (!msg->buf)
        goto final_error;

    return msg;

final_error:

    abcdk_comm_message_unref(&msg);

    return NULL;
}

int abcdk_comm_message_realloc(abcdk_comm_message_t *msg, size_t size)
{
    void *new_buf = NULL;

    assert(msg != NULL && size > 0);

    /*新的大小与旧的大小一样时，不需要调整。*/
    if (msg->size == size)
        goto final;

    msg->size = size;

    /*新的容量与旧的容量一样时，不需要调整。*/
    if (msg->capacity == ABCDK_MAX(msg->size, 1024UL))
        goto final;

    msg->capacity = ABCDK_MAX(msg->size, 1024UL);

    new_buf = abcdk_heap_realloc(msg->buf, msg->capacity);
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

void abcdk_comm_message_reset(abcdk_comm_message_t *msg)
{
    assert(msg != NULL);

    msg->offset = 0;
}

void *abcdk_comm_message_data(const abcdk_comm_message_t *msg)
{
    assert(msg != NULL);

    return msg->buf;
}

size_t abcdk_comm_message_size(const abcdk_comm_message_t *msg)
{
    assert(msg != NULL);

    return msg->size;
}

size_t abcdk_comm_message_offset(const abcdk_comm_message_t *msg)
{
    assert(msg != NULL);

    return msg->offset;
}

void abcdk_comm_message_protocol_set(abcdk_comm_message_t *msg, abcdk_comm_message_protocol_cb protocol_cb)
{
    assert(msg != NULL && protocol_cb != NULL);

    msg->protocol_cb = protocol_cb;
}

int abcdk_comm_message_recv(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
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
    if (msg->protocol_cb)
    {
        chk = msg->protocol_cb(node, msg);
        if (chk < 0)
            return -1;
        else if (chk == 0)
            goto MORE_DATA;
    }

    return 1;
}

int abcdk_comm_message_send(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
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
