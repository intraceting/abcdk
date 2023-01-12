/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/receiver.h"

/** 接收器对象。*/
struct _abcdk_receiver
{
    /** 引用计数器。*/
    volatile int refcount;
    
    /** 临时文件名。*/
    char tmp_file[PATH_MAX];
    
    /** 外部缓存对象。*/
    abcdk_object_t *tmp_obj;

    /** 内存块指针。*/
    void *buf;

    /* 读写偏移量。*/
    size_t offset;

    /** 容量。*/
    size_t capacity;

    /** 长度。*/
    size_t size;

    /** 消息协议。*/
    abcdk_receiver_protocol_t protocol;
    
};// abcdk_receiver_t;


void abcdk_receiver_unref(abcdk_receiver_t **msg)
{
    abcdk_receiver_t *msg_p = NULL;

    if (!msg || !*msg)
        return;

    msg_p = *msg;
    *msg = NULL;

    if (abcdk_atomic_fetch_and_add(&msg_p->refcount, -1) != 1)
        return;

    assert(msg_p->refcount == 0);

    /*创建时，如果绑定外部内部对象，则内部对象没有创建内存。*/
    if(msg_p->tmp_obj)
        abcdk_object_unref(&msg_p->tmp_obj);
    else
        abcdk_heap_free2(&msg_p->buf);

    if(access(msg_p->tmp_file,F_OK)==0)
        remove(msg_p->tmp_file);

    abcdk_heap_free(msg_p);
}

abcdk_receiver_t *abcdk_receiver_refer(abcdk_receiver_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_receiver_t *abcdk_receiver_alloc(const char *tempdir)
{
    abcdk_receiver_t *msg = NULL;
    
    msg = abcdk_heap_alloc(sizeof(abcdk_receiver_t));
    if (!msg)
        return NULL;

    msg->refcount = 1;
    msg->offset = 0;
    msg->tmp_obj = NULL;
    msg->size = 0;
    msg->capacity = 0;
    msg->buf = NULL;

    if (tempdir && *tempdir)
    {
        if (access(tempdir, W_OK) != 0)
            goto final_error;

        strncpy(msg->tmp_file, tempdir, PATH_MAX - 6);
        abcdk_dirdir(msg->tmp_file, "abcdk-receiver-XXXXXX");

        msg->tmp_obj = abcdk_mmap_tempfile(msg->tmp_file, 4096, 1, 1);
        if (!msg->tmp_obj)
            goto final_error;
    }

    return msg;

final_error:

    abcdk_receiver_unref(&msg);

    return NULL;
}

void *abcdk_receiver_data(const abcdk_receiver_t *msg)
{
    assert(msg != NULL);

    return msg->buf;
}

size_t abcdk_receiver_size(const abcdk_receiver_t *msg)
{
    assert(msg != NULL);

    return msg->size;
}

size_t abcdk_receiver_offset(const abcdk_receiver_t *msg)
{
    assert(msg != NULL);

    return msg->offset;
}

int abcdk_receiver_resize(abcdk_receiver_t *msg, size_t size)
{
    void *new_buf = NULL;
    int chk;

    assert(msg != NULL && size > 0);

    /*新的大小与旧的大小一样时，不需要调整。*/
    if (msg->size == size)
        goto final;

    msg->size = size;

#define ABCDK_RECEIVER_SIZE_DEFAULT (20*1024)

    /*新的容量与旧的容量一样时，不需要调整。*/
    if (msg->capacity == ABCDK_MAX(msg->size, ABCDK_RECEIVER_SIZE_DEFAULT))
        goto final;

    msg->capacity = ABCDK_MAX(msg->size, ABCDK_RECEIVER_SIZE_DEFAULT);

    if (msg->tmp_obj)
    {
        /*内存数据落盘。*/
        chk = abcdk_msync(msg->tmp_obj,0);
        if (chk != 0)
            return -1;

        /*重新映射文件。*/
        chk = abcdk_mremap(msg->tmp_obj, msg->capacity + 1, 1, 1);
        if (chk != 0)
            return -1;

        /*绑定新内存。*/
        msg->buf = msg->tmp_obj->pptrs[0];
    }
    else
    {
        /*重新申请内存。*/
        new_buf = abcdk_heap_realloc(msg->buf, msg->capacity + 1);
        if (!new_buf)
            return -1;

        /*绑定新内存。*/
        msg->buf = new_buf;
    }

    /*多出的一个字节赋值为0。*/
    ABCDK_PTR2U8(msg->buf, msg->capacity) = 0;

final:

    /*修正编移量。*/
    if (msg->offset > msg->size)
        msg->offset = msg->size;

    return 0;
}

void abcdk_receiver_protocol_set(abcdk_receiver_t *msg, abcdk_receiver_protocol_t *prot)
{
    assert(msg != NULL && prot != NULL);
    ABCDK_ASSERT(prot->unpack_cb != NULL,"未绑定解包回调函数，消息对象无法正常工作。");

    msg->protocol = *prot;
}

int abcdk_receiver_recv(abcdk_receiver_t *msg, const void *data,size_t size,size_t *remain)
{
    ssize_t rsize = 0;
    size_t rall = 0,diff = 0;
    int chk;

    assert(msg != NULL && data != NULL && size > 0 && remain != NULL);
    
    /*默认无剩余数据。*/
    *remain = 0;

    for (;;)
    {
        /*检测接收的数据是否完整。*/
        if (msg->protocol.unpack_cb)
        {
            diff = 0;
            chk = msg->protocol.unpack_cb(msg->protocol.opaque, msg, &diff);
        }
        else
        {
            diff = size - rall;
            chk = 0;
        }

        if (chk != 0)
            break;

        /*检查可用空间。*/
        if (msg->size - msg->offset < diff)
        {
            if (abcdk_receiver_resize(msg, msg->size + diff) != 0)
                chk = -1;
        }

        if (chk != 0)
            break;

        rsize = ABCDK_MIN(msg->size - msg->offset, size - rall);
        rsize = ABCDK_MIN(rsize,diff);
        if (rsize <= 0)
        {
            /*如果未指定解包回调函数，则未知的流数据已经接收完整。*/
            chk = (msg->protocol.unpack_cb ? 0 : 1);
            break;
        }
        else
        {
            memcpy(ABCDK_PTR2VPTR(msg->buf, msg->offset), ABCDK_PTR2VPTR(data, rall), rsize);
            msg->offset += rsize;
            rall += rsize;
        }
    }

    /*计算剩于数据长度。*/
    *remain = size - rall;

    return chk;
}
