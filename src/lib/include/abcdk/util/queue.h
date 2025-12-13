/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_QUEUE_H
#define ABCDK_UTIL_QUEUE_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"

__BEGIN_DECLS

/** 简单的队列.*/
typedef struct _abcdk_queue abcdk_queue_t;

/** 消息销毁回调函数.*/
typedef void (*abcdk_queue_msg_destroy_cb)(void *msg);

/**
 * 释放.
*/
void abcdk_queue_free(abcdk_queue_t **ctx);

/**
 * 创建.
 * 
 * @note 先进先出.
 * 
 * @param [in] cb 消息销毁回调函数.
*/
abcdk_queue_t *abcdk_queue_alloc(abcdk_queue_msg_destroy_cb cb);

/**长度.*/
uint64_t abcdk_queue_length(abcdk_queue_t *ctx);

/**解锁. */
void abcdk_queue_unlock(abcdk_queue_t *ctx);

/**加锁. */
void abcdk_queue_lock(abcdk_queue_t *ctx);

/**通知. */
void abcdk_queue_signal(abcdk_queue_t *ctx,int broadcast);

/**
 * 等待.
 * 
 * @param [in] timeout – 时长(毫秒).< 0 直到有事件或出错.
 * 
 * @return 0 成功(有事件), !0 失败(超时或出错).
 */
int abcdk_queue_wait(abcdk_queue_t *ctx, time_t timeout);

/**
 * 加入消息.
 * 
 * @note 消息对象将被托管, 在消息对象从队列中弹出之前, 应用层不可以继续访问消息对象.
 * 
 * @return 0 成功, -1 失败.
*/
int abcdk_queue_push(abcdk_queue_t *ctx, void *msg);

/**弹出消息.*/
void * abcdk_queue_pop(abcdk_queue_t *ctx);



__END_DECLS

#endif //ABCDK_UTIL_QUEUE_H