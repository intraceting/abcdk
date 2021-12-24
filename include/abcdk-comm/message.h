/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_MESSAGE_H
#define ABCDK_COMM_MESSAGE_H

#include "abcdk-comm/comm.h"

__BEGIN_DECLS

/** 消息对象。*/
typedef struct _abcdk_comm_message abcdk_comm_message_t;

/** 数据包协议回调函数。
 * 
 * @return 1 数据包完整，0 需要更多数据，-1 不支持的协议。
*/
typedef int (*abcdk_comm_message_protocol_cb)(abcdk_comm_node_t *node, abcdk_comm_message_t *msg);

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
 * 调整消息对象大小。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_comm_message_realloc(abcdk_comm_message_t *msg, size_t size);

/**
 * 重置。
*/
void abcdk_comm_message_reset(abcdk_comm_message_t *msg);

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
 * 设置数据包协议(接收有效)。
*/
void abcdk_comm_message_protocol_set(abcdk_comm_message_t *msg, abcdk_comm_message_protocol_cb protocol_cb);

/**
 * 接收消息。
 * 
 * @return 1 缓存区已满，0 缓存区未满。
*/
int abcdk_comm_message_recv(abcdk_comm_node_t *node, abcdk_comm_message_t *msg);

/**
 * 发送消息。
 * 
 * @return 1 发送完毕，0 有未发送数据。
*/
int abcdk_comm_message_send(abcdk_comm_node_t *node, abcdk_comm_message_t *msg);


__END_DECLS

#endif //ABCDK_COMM_MESSAGE_H