/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_QUEUE_H
#define ABCDK_UTIL_QUEUE_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"

__BEGIN_DECLS

/** 简单的队列。*/
typedef struct _abcdk_queue abcdk_queue_t;

/** 
 * 消息销毁回调函数。
*/
typedef void (*abcdk_queue_msg_destroy_cb)(void *msg);

/**
 * 释放。
*/
void abcdk_queue_free(abcdk_queue_t **queue);

/**
 * 创建。
 * 
 * @param [in] cb 消息销毁回调函数。
*/
abcdk_queue_t *abcdk_queue_alloc(abcdk_queue_msg_destroy_cb cb);

/**
 * 获取队列长度。
*/
size_t abcdk_queue_count(abcdk_queue_t *queue);

/**
 * 向队列中加入消息。
 * 
 * @note 消息对象将被托管，在消息对象从队列中弹出之前，应用层不可以继续访问消息对象。
 * 
 * @param [in] first !0 头部，0 尾部。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_queue_push(abcdk_queue_t *queue, const void *msg, int first);

/**
 * 从队列中弹出消息。
 * 
 * @param [in] first !0 头部，0 尾部。
 * 
 * @return !NULL(0) 成功(消息对象指针)，NULL(0) 失败(队列为空)。
*/
const void *abcdk_queue_pop(abcdk_queue_t *queue, int first);

/**
 * 等待。
 * 
 * @param [in] timeout 超时(毫秒)。
 * 
 * @return 0 有事件，!0 超时或出错。
*/
int abcdk_queue_wait(abcdk_queue_t *queue,time_t timeout);


__END_DECLS

#endif //ABCDK_UTIL_QUEUE_H