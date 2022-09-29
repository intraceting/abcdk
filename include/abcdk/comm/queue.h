/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_QUEUE_H
#define ABCDK_COMM_QUEUE_H

#include "abcdk/comm/comm.h"
#include "abcdk/comm/message.h"

__BEGIN_DECLS

/** 消息队列。*/
typedef struct _abcdk_comm_queue abcdk_comm_queue_t;

/**
 * 释放消息队列。
*/
void abcdk_comm_queue_free(abcdk_comm_queue_t **queue);

/**
 * 创建消息队列。
*/
abcdk_comm_queue_t *abcdk_comm_queue_alloc();

/**
 * 消息队列长度。
*/
size_t abcdk_comm_queue_count(abcdk_comm_queue_t *queue);

/**
 * 消息加入到队列末尾。
 * 
 * @warning 消息对象将被托管，在消息对象从队列中弹出之前，应用层不可以继续访问消息对象。
 * @warning 不会改变消息对象的引用计数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_comm_queue_push(abcdk_comm_queue_t *queue, abcdk_comm_message_t *msg);

/**
 * 消息从队列中弹出。
 * 
 * @warning 不会改变消息对象的引用计数。
 * 
 * @return !NULL(0) 成功(消息对象指针)，NULL(0) 失败(队列为空)。
*/
abcdk_comm_message_t *abcdk_comm_queue_pop(abcdk_comm_queue_t *queue);


__END_DECLS

#endif //ABCDK_COMM_QUEUE_H