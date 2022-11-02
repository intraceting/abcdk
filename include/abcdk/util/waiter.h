/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_WAITER_H
#define ABCDK_UTIL_WAITER_H

#include "abcdk/util/general.h"
#include "abcdk/util/time.h"
#include "abcdk/util/map.h"
#include "abcdk/util/queue.h"

__BEGIN_DECLS

/** 消息服务员。*/
typedef struct _abcdk_waiter abcdk_waiter_t;

/**
 * 释放消息服务员。
*/
void abcdk_waiter_free(abcdk_waiter_t **waiter);

/** 
 * 创建消息服务员。
*/
abcdk_waiter_t *abcdk_waiter_alloc();

/**
 * 消息请求(注册)。
 * 
 * @warning 消息队列将被托管理，应用层不可以继续访问消息对象。
 * 
 * @return 0 成功，-1 失败(KEY重复)。
*/
int abcdk_waiter_request(abcdk_waiter_t *waiter,uint64_t key, abcdk_queue_t *queue);

/**
 * 等待消息。
 * 
 * @param max 最大应答消息数量。
 * @param timeout 超时(毫秒)。
 * 
 * @return !NULL(0) 成功(消息队列指针)，NULL(0) 失败(KEY不存在)。
*/
abcdk_queue_t *abcdk_waiter_wait(abcdk_waiter_t *waiter,uint64_t key, size_t max, time_t timeout);

/**
 * 消息应答。
 * 
 * @warning 消息将被托管理，应用层(应答者)不可以继续访问消息对象。
 * 
 * @return 0 成功，-1 失败(KEY不存在)。
*/
int abcdk_waiter_response(abcdk_waiter_t *waiter,uint64_t key, const void *msg);

/** 
 * 取消(仅影响等待)。
*/
void abcdk_waiter_cancel(abcdk_waiter_t *waiter);

/** 
 * 恢复(仅影响等待)。
*/
void abcdk_waiter_resume(abcdk_waiter_t *waiter);

__END_DECLS

#endif //ABCDK_UTIL_WAITER_H