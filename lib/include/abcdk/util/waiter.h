/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_WAITER_H
#define ABCDK_UTIL_WAITER_H

#include "abcdk/util/general.h"
#include "abcdk/util/time.h"
#include "abcdk/util/map.h"
#include "abcdk/util/mutex.h"

__BEGIN_DECLS

/**服务员。*/
typedef struct _abcdk_waiter abcdk_waiter_t;

/** 消息销毁回调函数。*/
typedef void (*abcdk_waiter_msg_destroy_cb)(void *msg);

/**
 * 释放。
*/
void abcdk_waiter_free(abcdk_waiter_t **waiter);

/** 
 * 创建。
 * 
 * @param [in] cb 消息销毁回调函数。
*/
abcdk_waiter_t *abcdk_waiter_alloc(abcdk_waiter_msg_destroy_cb cb);

/**
 * 注册。
 * 
 * @return 0 成功，-1 失败(KEY重复)。
*/
int abcdk_waiter_register(abcdk_waiter_t *waiter,uint64_t key);

/**
 * 等待。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return !NULL(0) 成功(对象指针)，NULL(0) 失败(超时或KEY不存在)。
*/
void *abcdk_waiter_wait(abcdk_waiter_t *waiter,uint64_t key,time_t timeout);

/**
 * 应答。
 * 
 * @note 将被托管理，应用层不可以继续访问应答对象。
 * 
 * @return 0 成功，-1 失败(KEY不存在)。
*/
int abcdk_waiter_response(abcdk_waiter_t *waiter,uint64_t key, void *msg);

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