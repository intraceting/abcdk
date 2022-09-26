/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_MESSAGE_H
#define ABCDK_COMM_MESSAGE_H

#include "util/mmap.h"
#include "comm/comm.h"

__BEGIN_DECLS

/** 消息对象。*/
typedef struct _abcdk_comm_message abcdk_comm_message_t;

/**消息协议。*/
typedef struct _abcdk_comm_message_protocol
{
    /** 环境指针。*/
    void *opaque;

    /**
     * 消息解包回调函数。
     *
     * @return 1 消息包完整，0 需要更多数据，-1 不支持的协议(或有错发生)。
     */
    int (*unpack_cb)(void *opaque, abcdk_comm_message_t *msg);

} abcdk_comm_message_protocol_t;

/**
 * 减少对象的引用计数。
 * 
 * @warning 当引用计数为0时，对像将被删除。
*/
void abcdk_comm_message_unref(abcdk_comm_message_t **msg);

/**
 * 增加对象的引用计数。
*/
abcdk_comm_message_t *abcdk_comm_message_refer(abcdk_comm_message_t *src);

/**
 * 创建消息对象。
*/
abcdk_comm_message_t *abcdk_comm_message_alloc(size_t size);

/**
 * 创建消息对象。
 * 
 * @warning 内存对象将被托管，应用层不可以继续访问内存对象。
 * 
 * @param [in] obj 内存对象指针，索引为0号元素有效。注：仅做指针复制，不会改变对象的引用计数。
*/
abcdk_comm_message_t *abcdk_comm_message_alloc2(abcdk_object_t *obj);

/**
 * 创建消息对象。
 * 
 * @param name 文件名(或全路径)的指针。
 * @param [in] truncate 截断文件(或扩展文件)。0 忽略。
 * @param [in] rw !0 读写，0 只读。
 * 
 * @return NULL(0) 失败，!NULL(0) 成功。
*/
abcdk_comm_message_t* abcdk_comm_message_alloc3(const char* name,size_t truncate,int rw);

/**
 * 调整消息对象大小。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_comm_message_realloc(abcdk_comm_message_t *msg, size_t size);

/**
 * 扩展消息对象大小。
 * 
 * @param [in] size 增量。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_comm_message_expand(abcdk_comm_message_t *msg, size_t size);

/**
 * 重置读写偏移量。
*/
void abcdk_comm_message_reset(abcdk_comm_message_t *msg,size_t offset);

/**
 * 获取数据区指针。
*/
void *abcdk_comm_message_data(const abcdk_comm_message_t *msg);

/**
 * 获取数据区长度。
*/
size_t abcdk_comm_message_size(const abcdk_comm_message_t *msg);

/**
 * 获取读写偏移量。
*/
size_t abcdk_comm_message_offset(const abcdk_comm_message_t *msg);

/**
 * 排出已读写的数据，同时重置偏移量。
*/
void abcdk_comm_message_drain(abcdk_comm_message_t *msg,size_t size);

/**
 * 发送消息。
 * 
 * @return 1 发送完毕，0 有未发送数据。
*/
int abcdk_comm_message_send(abcdk_comm_node_t *node, abcdk_comm_message_t *msg);

/**
 * 设置数据包协议(接收有效)。
*/
void abcdk_comm_message_protocol_set(abcdk_comm_message_t *msg, abcdk_comm_message_protocol_t *prot);

/**
 * 接收消息。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_comm_message_recv(abcdk_comm_node_t *node, abcdk_comm_message_t *msg);

/**
 * 接收消息(从缓存)。
 * 
 * @param [in out] remain 缓存剩余的数据长度，返回时填充。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_comm_message_recv2(const void *data,size_t size,size_t *remain, abcdk_comm_message_t *msg);

__END_DECLS

#endif //ABCDK_COMM_MESSAGE_H
