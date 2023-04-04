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

/** 服务员。*/
typedef struct _abcdk_waiter abcdk_waiter_t;

/**
 * 释放。
*/
void abcdk_waiter_free(abcdk_waiter_t **waiter);

/** 
 * 创建。
*/
abcdk_waiter_t *abcdk_waiter_alloc();

/**
 * 请求(注册)。
 * 
 * @note 队列将被托管理，应用层不可以继续访问对象。
 * 
 * @return 0 成功，-1 失败(KEY重复)。
*/
int abcdk_waiter_request(abcdk_waiter_t *waiter,uint64_t key, abcdk_queue_t *queue);

/**
 * 等待。
 * 
 * @param max 最大应答数量。
 * @param timeout 超时(毫秒)。
 * 
 * @return !NULL(0) 成功(队列指针)，NULL(0) 失败(KEY不存在)。
*/
abcdk_queue_t *abcdk_waiter_wait(abcdk_waiter_t *waiter,uint64_t key, size_t max, time_t timeout);

/**
 * 应答。
 * 
 * @note 将被托管理，应用层不可以继续访问应答对象。
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