/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_MESSAGE_H
#define ABCDK_COMM_MESSAGE_H

#include "abcdk/util/path.h"
#include "abcdk/util/mmap.h"
#include "abcdk/comm/comm.h"

__BEGIN_DECLS

/** 消息对象。*/
typedef struct _abcdk_message abcdk_message_t;

/**消息协议。*/
typedef struct _abcdk_message_protocol
{
    /** 环境指针。*/
    void *opaque;

    /**
     * 消息解包回调函数。
     * 
     * @param [out] diff 差额(待增量)。返回时填写。
     *
     * @return 1 消息包完整，0 需要更多数据，-1 不支持的协议(或有错发生)。
     */
    int (*unpack_cb)(void *opaque, abcdk_message_t *msg,size_t *diff);

} abcdk_message_protocol_t;

/**
 * 减少对象的引用计数。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_message_unref(abcdk_message_t **msg);

/**
 * 增加对象的引用计数。
*/
abcdk_message_t *abcdk_message_refer(abcdk_message_t *src);

/**
 * 创建消息对象。
 * 
 * @param [in] tempdir 缓存目录。NULL(0) 忽略。
*/
abcdk_message_t *abcdk_message_alloc(const char *tempdir);

/**
 * 获取数据区指针。
*/
void *abcdk_message_data(const abcdk_message_t *msg);

/**
 * 获取数据区长度。
*/
size_t abcdk_message_size(const abcdk_message_t *msg);

/**
 * 获取读写偏移量。
*/
size_t abcdk_message_offset(const abcdk_message_t *msg);

/**
 * 调整消息对象大小。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_message_resize(abcdk_message_t *msg, size_t size);

/**
 * 设置数据包协议。
*/
void abcdk_message_protocol_set(abcdk_message_t *msg, abcdk_message_protocol_t *prot);

#define abcdk_message_protocol_set_simple(msg, op, cb) \
    {                                                  \
        abcdk_message_protocol_t prot = {0};           \
        prot.opaque = op;                              \
        prot.unpack_cb = cb;                           \
        abcdk_message_protocol_set(msg, &prot);        \
    }

/**
 * 接收消息(从缓存)。
 * 
 * @param [in out] remain 缓存剩余的数据长度，返回时填充。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_message_recv(abcdk_message_t *msg,const void *data,size_t size,size_t *remain);


__END_DECLS

#endif //ABCDK_COMM_MESSAGE_H
