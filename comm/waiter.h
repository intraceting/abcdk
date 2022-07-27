/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_WAITER_H
#define ABCDK_COMM_WAITER_H

#include "comm/comm.h"
#include "comm/message.h"
#include "comm/queue.h"

__BEGIN_DECLS

/** 消息服务员。*/
typedef struct _abcdk_comm_waiter abcdk_comm_waiter_t;

/** 
 * 消息KEY比较函数。
 * 
 * @return 0 key1等于key2，!0 key1不等于key2。
*/
typedef int (*abcdk_comm_waiter_compare_cb)(const void *key1, size_t size1, const void *key2, size_t size2);

/**
 * 释放消息服务员。
*/
void abcdk_comm_waiter_free(abcdk_comm_waiter_t **waiter);

/** 
 * 创建消息服务员。
*/
abcdk_comm_waiter_t *abcdk_comm_waiter_alloc();

/**
 * 设置比较回调函数。
*/
void abcdk_comm_waiter_set_compare_callback(abcdk_comm_waiter_t *waiter,
                                            abcdk_comm_waiter_compare_cb compare_cb);

/**
 * 消息请求(注册)。
 * 
 * @return 0 成功，-1 失败(KEY重复)。
*/
int abcdk_comm_waiter_request(abcdk_comm_waiter_t *waiter,
                              const void *key, size_t ksize);

/** 消息请求(注册)。*/
#define abcdk_comm_waiter_request2(waiter, key) \
    abcdk_comm_waiter_request((waiter), (key), sizeof(*(key)))

/**
 * 等待消息。
 * 
 * @param max 最大应答消息数量。
 * @param timeout 超时(毫秒)。
 * 
 * @return !NULL(0) 成功(消息队列指针)，NULL(0) 失败(KEY不存在)。
*/
abcdk_comm_queue_t *abcdk_comm_waiter_wait(abcdk_comm_waiter_t *waiter,
                                           const void *key, size_t ksize,
                                           size_t max, time_t timeout);

/** 等待消息。*/
#define abcdk_comm_waiter_wait2(waiter, key, max, timeout) \
    abcdk_comm_waiter_wait((waiter), (key), sizeof(*(key)), (max), (timeout))

/**
 * 消息应答。
 * 
 * @warning 消息将被托管理，应用层(应答者)不可以继续访问消息对象。
 * 
 * @return 0 成功，-1 失败(KEY不存在)。
*/
int abcdk_comm_waiter_response(abcdk_comm_waiter_t *waiter,
                               const void *key, size_t ksize,
                               abcdk_comm_message_t *msg);

/** 消息应答。*/
#define abcdk_comm_waiter_response2(waiter, key, msg) \
    abcdk_comm_waiter_response((waiter), (key), sizeof(*(key)), (msg))

/** 
 * 取消(仅影响等待)。
*/
void abcdk_comm_waiter_cancel(abcdk_comm_waiter_t *waiter);

/** 
 * 恢复(仅影响等待)。
*/
void abcdk_comm_waiter_resume(abcdk_comm_waiter_t *waiter);

__END_DECLS

#endif //ABCDK_COMM_WAITER_H