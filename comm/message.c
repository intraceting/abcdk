/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-comm/message.h"

/*
 * 消息格式。
 *
 * ------------------------------------------
 * |Message Data                            |
 * ------------------------------------------
 * |Length  |Flag    |Message ID |Customer  |
 * |4 Bytes |4 Bytes |8 Bytes    |N Bytes   |
 * ------------------------------------------
*/

#define ABCDK_COMM_MSG_HDR_SIZE             16
#define ABCDK_COMM_MSG_MAX_SIZE             (256*1024*1024-1)

/***/
typedef struct _abcdk_comm_msg
{
    /** 缓存区指针。*/
    void *buf;

    /** 缓存区容量。*/
    size_t capacity;

    /** 消息长度。*/
    uint32_t size;

    /** 消息标志。*/
    uint32_t flag; 

    /** 消息编号。*/
    uint32_t mid;

    /* 读写偏移量。*/
    size_t offset;

}abcdk_comm_msg_t;


#define ABCDK_COMM_MSG_FIELD_LEN(msg)       ABCDK_PTR2U32((msg)->buf, 0)
#define ABCDK_COMM_MSG_FIELD_FLAG(msg)      ABCDK_PTR2U32((msg)->buf, 4)
#define ABCDK_COMM_MSG_FIELD_MID(msg)       ABCDK_PTR2U64((msg)->buf, 8)
#define ABCDK_COMM_MSG_FIELD_CUST(msg)      ABCDK_PTR2U8PTR((msg)->buf, ABCDK_COMM_MSG_HDR_SIZE)

void abcdk_comm_msg_free(abcdk_comm_msg_t **msg)
{
    abcdk_comm_msg_t *msg_p = NULL;

    if(!msg || !*msg)
        return;

    msg_p = *msg;

    if(msg_p->buf)
        abcdk_heap_free2(&msg_p->buf);
    
    abcdk_heap_free2((void**)msg);
}

abcdk_comm_msg_t *abcdk_comm_msg_alloc(size_t size)
{
    abcdk_comm_msg_t *msg = NULL;

    assert(size > 0 && size <= ABCDK_COMM_MSG_MAX_SIZE);
    
    msg = abcdk_heap_alloc(sizeof(abcdk_comm_msg_t));
    if (!msg)
        goto final_error;
    
    msg->offset = 0;
    msg->size = ABCDK_COMM_MSG_HDR_SIZE + size;
    msg->capacity = ABCDK_MAX(msg->size,1024UL);
    msg->buf = abcdk_heap_alloc(msg->capacity);

    if (!msg->buf)
        goto final_error;

    return msg;

final_error:

    abcdk_comm_msg_free(&msg);

    return NULL;
}

int abcdk_comm_msg_realloc(abcdk_comm_msg_t *msg,size_t size)
{
    void * new_buf = NULL;

    assert(msg != NULL && size > 0 && size <= ABCDK_COMM_MSG_MAX_SIZE);

    msg->size = ABCDK_COMM_MSG_HDR_SIZE + size;
    msg->capacity = ABCDK_MAX(msg->size,1024UL);

    new_buf = abcdk_heap_realloc(msg->buf,msg->capacity);
    if (!new_buf)
        return -1;

    /*绑定新内存。*/
    msg->buf = new_buf;

    /*修正编移量。*/
    if(msg->offset > msg->size)
        msg->offset = msg->size;

    return 0;
}

uint32_t abcdk_comm_msg_flag(abcdk_comm_msg_t *msg)
{
    assert(msg != NULL);

    return msg->flag;
}

uint32_t abcdk_comm_msg_flag_set(abcdk_comm_msg_t *msg, uint32_t flag)
{
    uint32_t old;

    assert(msg != NULL);

    old = msg->flag;
    msg->flag = flag;

    return old;
}

uint64_t abcdk_comm_msg_mid(abcdk_comm_msg_t *msg)
{
    assert(msg != NULL);

    return msg->flag;
}

uint64_t abcdk_comm_msg_mid_set(abcdk_comm_msg_t *msg, uint64_t mid)
{
    uint64_t old;

    assert(msg != NULL);

    old = msg->mid;
    msg->mid = mid;

    return old;
}

void *abcdk_comm_msg_data(const abcdk_comm_msg_t *msg)
{
    assert(msg != NULL);

    return ABCDK_COMM_MSG_FIELD_CUST(msg);
}

size_t abcdk_comm_msg_size(const abcdk_comm_msg_t *msg)
{
    assert(msg != NULL);

    return msg->size - ABCDK_COMM_MSG_HDR_SIZE;
}

int abcdk_comm_msg_recv(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg)
{
    uint32_t size;
    ssize_t rsize;
    int chk;

    assert(node != NULL && msg != NULL);

    if (msg->offset < ABCDK_COMM_MSG_HDR_SIZE)
    {
        rsize = abcdk_comm_read(node, ABCDK_PTR2VPTR(msg->buf, msg->offset), ABCDK_COMM_MSG_HDR_SIZE - msg->offset);
        if (rsize == 0)
            return -1;
        else if (rsize < 0)
            return 0;
        else if (rsize > 0)
            msg->offset += rsize;

        return abcdk_comm_msg_recv(node,msg);
    }
    else 
    {
        size = abcdk_endian_b_to_h32(ABCDK_COMM_MSG_FIELD_LEN(msg));
        if(size - ABCDK_COMM_MSG_HDR_SIZE > ABCDK_COMM_MSG_MAX_SIZE)
            return -1;

        if(msg->size != size)
        {
            chk = abcdk_comm_msg_realloc(msg,size);
            if(chk != 0)
                return -1;
        }

        rsize = abcdk_comm_read(node, ABCDK_PTR2VPTR(msg->buf, msg->offset), msg->size - msg->offset);
        if (rsize == 0)
            return -1;
        else if (rsize < 0)
            return 0;
        else if (rsize > 0)
            msg->offset += rsize;
    }

    /*检测接收的数据是否完整。*/
    if(msg->size != msg->offset)
        return 0;

    /*转换部分字段的字节序。*/
    msg->flag = abcdk_endian_b_to_h32(ABCDK_COMM_MSG_FIELD_FLAG(msg));
    msg->mid = abcdk_endian_b_to_h32(ABCDK_COMM_MSG_FIELD_MID(msg));
    
    return 1;
}

int abcdk_comm_msg_send(abcdk_comm_node_t *node, abcdk_comm_msg_t *msg)
{
    uint32_t size;
    ssize_t wsize;
    int chk;

    assert(node != NULL && msg != NULL);

    if (msg->offset < 1)
    {
        /*转换部分字段的字节序。*/
        ABCDK_COMM_MSG_FIELD_LEN(msg) = abcdk_endian_h_to_b32(msg->size);
        ABCDK_COMM_MSG_FIELD_FLAG(msg) = abcdk_endian_h_to_b32(msg->flag);
        ABCDK_COMM_MSG_FIELD_MID(msg) = abcdk_endian_h_to_b32(msg->mid);
    }

    wsize = abcdk_comm_write(node, ABCDK_PTR2VPTR(msg->buf, msg->offset), msg->size - msg->offset);
    if (wsize == 0)
        return -1;
    else if (wsize < 0)
        return 0;
    else if (wsize > 0)
        msg->offset += wsize;

    /*检测发送的数据是否完整。*/
    if (msg->size == msg->offset)
        return 1;

    return 0;
}