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
    
    /** 内存对象指针。*/
    abcdk_object_t *user_obj;

    /** 内存块指针。*/
    void *buf;

    /* 读写偏移量。*/
    size_t offset;

    /** 容量。*/
    size_t capacity;

    /** 长度。*/
    size_t size;

    /** 消息协议。*/
    abcdk_comm_message_protocol_t protocol;
    
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

    /*创建时，如果绑定外部内部对象，则内部对象没有创建内存。*/
    if(msg_p->user_obj)
        abcdk_object_unref(&msg_p->user_obj);
    else
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
    msg->user_obj = NULL;
    msg->size = size;
    msg->capacity = ABCDK_MAX(msg->size, 4096UL);
    msg->buf = abcdk_heap_alloc(msg->capacity + 1);

    if (!msg->buf)
        goto final_error;

    return msg;

final_error:

    abcdk_comm_message_unref(&msg);

    return NULL;
}

abcdk_comm_message_t *abcdk_comm_message_alloc2(abcdk_object_t *obj)
{
    abcdk_comm_message_t *msg = NULL;

    assert(obj != NULL && obj->pptrs[0] != NULL && obj->sizes[0] > 0);

    msg = abcdk_heap_alloc(sizeof(abcdk_comm_message_t));
    if (!msg)
        return NULL;

    msg->refcount = 1;
    msg->offset = 0;
    msg->user_obj = obj;
    msg->size = msg->user_obj->sizes[0];
    msg->capacity = msg->user_obj->sizes[0];
    msg->buf = msg->user_obj->pptrs[0];

    return msg;
}

int abcdk_comm_message_realloc(abcdk_comm_message_t *msg, size_t size)
{
    void *new_buf = NULL;

    assert(msg != NULL && size > 0);

    ABCDK_ASSERT(msg->user_obj == NULL,"绑定外部内部对象时，不支持重新调整大小。");

    /*新的大小与旧的大小一样时，不需要调整。*/
    if (msg->size == size)
        goto final;

    msg->size = size;

    /*新的容量与旧的容量一样时，不需要调整。*/
    if (msg->capacity == ABCDK_MAX(msg->size, 4096UL))
        goto final;

    msg->capacity = ABCDK_MAX(msg->size, 4096UL);

    new_buf = abcdk_heap_realloc(msg->buf, msg->capacity + 1);
    if (!new_buf)
        return -1;

    /*多出的一个字节赋值为0。*/
    ABCDK_PTR2U8(new_buf,msg->capacity) = 0;

    /*绑定新内存。*/
    msg->buf = new_buf;

final:

    /*修正编移量。*/
    if (msg->offset > msg->size)
        msg->offset = msg->size;

    return 0;
}

int abcdk_comm_message_expand(abcdk_comm_message_t *msg, size_t size)
{
    assert(msg != NULL);

    return abcdk_comm_message_realloc(msg, abcdk_comm_message_size(msg) + size);
}

void abcdk_comm_message_reset(abcdk_comm_message_t *msg,size_t offset)
{
    assert(msg != NULL);

    msg->offset = offset;
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

void abcdk_comm_message_drain(abcdk_comm_message_t *msg)
{
    size_t remain;

    assert(msg != NULL);

    remain = msg->size - msg->offset;
    memmove(msg->buf,ABCDK_PTR2VPTR(msg->buf,msg->offset),remain);
    msg->offset = remain;
}

int abcdk_comm_message_send(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
{
    uint32_t size;
    ssize_t wsize;
    int chk;

    assert(node != NULL && msg != NULL);

    if (msg->offset >= msg->size)
        return 1;

    wsize = abcdk_comm_send(node, ABCDK_PTR2VPTR(msg->buf, msg->offset), msg->size - msg->offset);
    if (wsize <= 0)
        return 0;
    else if (wsize > 0)
        msg->offset += wsize;

    /*检测发送的数据是否完整。*/
    if (msg->size != msg->offset)
        return 0;

    return 1;
}

void abcdk_comm_message_protocol_set(abcdk_comm_message_t *msg, abcdk_comm_message_protocol_t *prot)
{
    assert(msg != NULL && prot != NULL);
    ABCDK_ASSERT(prot->unpack_cb != NULL,"未绑定解包回调函数，消息对象无法正常工作。");

    msg->protocol = *prot;
}

int abcdk_comm_message_recv(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
{
    ssize_t rsize = 0;
    int chk = 0;

    assert(node != NULL && msg != NULL);

MORE_DATA:

    rsize = abcdk_comm_recv(node, ABCDK_PTR2VPTR(msg->buf, msg->offset), msg->size - msg->offset);
    if (rsize <= 0)
        return 0;
    else
        msg->offset += rsize;

    /*检测接收的数据是否完整。*/
    if (msg->protocol.unpack_cb)
    {
        chk = msg->protocol.unpack_cb(msg->protocol.opaque, msg);
        if (chk < 0)
            return -1;
        else if (chk == 0)
            goto MORE_DATA;
    }

    return 1;
}

int abcdk_comm_message_recv2(const void *data,size_t size,size_t *remain, abcdk_comm_message_t *msg)
{
    ssize_t rsize = 0;
    size_t rall = 0;
    int chk;

    assert(data != NULL && size > 0 && remain != NULL && msg != NULL);

MORE_DATA:

    rsize = ABCDK_MIN(msg->size - msg->offset, size - rall);
    if (rsize <= 0)
    {
        chk = 0;
        goto FINAL_END;
    }
    else
    {
        memcpy(ABCDK_PTR2VPTR(msg->buf, msg->offset), ABCDK_PTR2VPTR(data, rall), rsize);
        msg->offset += rsize;
        rall += rsize;
    }

    /*检测接收的数据是否完整。*/
    if (msg->protocol.unpack_cb)
    {
        chk = msg->protocol.unpack_cb(msg->protocol.opaque, msg);
        if (chk < 0)
        {
            chk = -1;
            goto FINAL_END;
        }
        else if (chk == 0)
        {
            goto MORE_DATA;
        }
    }

    chk = 1;

FINAL_END:

    /*计算剩于数据长度。*/
    *remain = size - rall;

    return chk;
}