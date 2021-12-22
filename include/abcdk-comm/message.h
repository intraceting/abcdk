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

/** 消息缓存对象。*/
typedef struct _abcdk_comm_msg abcdk_comm_msg_t;

/** 数据包协议回调函数。
 * 
 * @return 1 数据包完整，0 需要更多数据。
*/
typedef int (*abcdk_comm_msg_protocol_cb)(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg);

/**
 * 释放消息缓存对象。
*/
void abcdk_comm_msg_unref(abcdk_comm_msg_t **msg);

/**
 * 息缓存对象增加引用。
 * 
 * @return !NULL(0) 成功(对象的指针)，NULL(0) 失败。
*/
abcdk_comm_msg_t *abcdk_comm_msg_refer(abcdk_comm_msg_t *src);

/**
 * 申请消息缓存对象。
*/
abcdk_comm_msg_t *abcdk_comm_msg_alloc(size_t size);

/**
 * 调整消息缓存大小。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_comm_msg_realloc(abcdk_comm_msg_t *msg, size_t size);

/**
 * 重置。
*/
void abcdk_comm_msg_reset(abcdk_comm_msg_t *msg);

/**
 * 获取数据区指针。
*/
void *abcdk_comm_msg_data(const abcdk_comm_msg_t *msg);

/**
 * 获取数据区长度。
*/
size_t abcdk_comm_msg_size(const abcdk_comm_msg_t *msg);

/**
 * 获取读写偏移量。
*/
size_t abcdk_comm_msg_offset(const abcdk_comm_msg_t *msg);

/**
 * 设置数据包协议。
*/
void abcdk_comm_msg_protocol_set(abcdk_comm_msg_t *msg,abcdk_comm_msg_protocol_cb protocol_cb);

/**
 * 接收消息。
 * 
 * @return 1 消息完整，0 消息不完整。
*/
int abcdk_comm_msg_recv(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg);

/**
 * 发送消息。
 * 
 * @return 1 消息完整，0 消息不完整。
*/
int abcdk_comm_msg_send(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg);



__END_DECLS

#endif //ABCDK_COMM_MESSAGE_H