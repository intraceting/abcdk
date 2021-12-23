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
typedef struct _abcdk_comm_msg abcdk_comm_msg_t;

/** 消息队列。*/
typedef struct _abcdk_comm_msg_queue abcdk_comm_msg_queue_t;

/** 数据包协议回调函数。
 * 
 * @return 1 数据包完整，0 需要更多数据。
*/
typedef int (*abcdk_comm_msg_protocol_cb)(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg);

/**
 * 释放消息对象。
*/
void abcdk_comm_msg_unref(abcdk_comm_msg_t **msg);

/**
 * 消息对象增加引用。
 * 
 * @return !NULL(0) 成功(对象的指针)，NULL(0) 失败。
*/
abcdk_comm_msg_t *abcdk_comm_msg_refer(abcdk_comm_msg_t *src);

/**
 * 创建消息对象。
*/
abcdk_comm_msg_t *abcdk_comm_msg_alloc(size_t size);

/**
 * 调整消息对象大小。
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
 * 设置数据包协议(接收有效)。
*/
void abcdk_comm_msg_protocol_set(abcdk_comm_msg_t *msg,abcdk_comm_msg_protocol_cb protocol_cb);

/**
 * 接收消息。
 * 
 * @return 1 缓存区已满，0 缓存区未满。
*/
int abcdk_comm_msg_recv(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg);

/**
 * 发送消息。
 * 
 * @return 1 发送完毕，0 有未发送数据。
*/
int abcdk_comm_msg_send(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg);


/**
 * 释放消息队列。
*/
void abcdk_comm_msg_queue_free(abcdk_comm_msg_queue_t **queue);

/**
 * 创建消息队列。
*/
abcdk_comm_msg_queue_t *abcdk_comm_msg_queue_alloc();

/**
 * 消息加入到队列末尾。
 * 
 * @warning 消息对象将被托管，在消息对象从队列中弹出之前，应用层不可以继续访问消息对象。
 * @warning 不会改变消息对象的引用计数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_comm_msg_queue_push(abcdk_comm_msg_queue_t *queue, abcdk_comm_msg_t *msg);

/**
 * 消息从队列中弹出。
 * 
 * @warning 不会改变消息对象的引用计数。
 * 
 * @return !NULL(0) 成功(消息对象指针)，NULL(0) 失败(队列为空)。
*/
abcdk_comm_msg_t *abcdk_comm_msg_queue_pop(abcdk_comm_msg_queue_t *queue);


__END_DECLS

#endif //ABCDK_COMM_MESSAGE_H