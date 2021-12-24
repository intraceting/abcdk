/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_EASY_H
#define ABCDK_COMM_EASY_H

#include "abcdk-comm/comm.h"
#include "abcdk-comm/message.h"
#include "abcdk-comm/queue.h"
#include "abcdk-comm/waiter.h"

__BEGIN_DECLS

/** 简单的通信对象。*/
typedef struct _abcdk_comm_easy abcdk_comm_easy_t;

/** 请求回调函数。*/
typedef void (*abcdk_comm_easy_request_cb)(abcdk_comm_easy_t *easy, abcdk_comm_message_t *req, abcdk_comm_message_t **rsp);

/**
 * 减少对象的引用计数。
 * 
 * @warning 当引用计数为0时，对像将被删除。
*/
void abcdk_comm_easy_unref(abcdk_comm_easy_t **easy);

/**
 * 增加对象的引用计数。
*/
abcdk_comm_easy_t *abcdk_comm_easy_refer(abcdk_comm_easy_t *src);

/**
 * 设置超时。
 * 
 * @warning 1、看门狗精度为200毫秒；2、超时生效时间受引擎的工作周期影响。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_easy_set_timeout(abcdk_comm_easy_t *easy, time_t timeout);

/**
 * 获取本机地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_easy_get_sockname(abcdk_comm_easy_t *easy, abcdk_sockaddr_t *addr);

/**
 * 获取远端地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_easy_get_peername(abcdk_comm_easy_t *easy, abcdk_sockaddr_t *addr);

/**
 * 设置应用层环境指针。
 * 
 * @return 旧的指针。
*/
void *abcdk_comm_easy_set_userdata(abcdk_comm_easy_t *easy, void *opaque);

/**
 * 获取应用层环境指针。
 * 
 * @return 旧的指针。
*/
void *abcdk_comm_easy_get_userdata(abcdk_comm_easy_t *easy);

/** 
 * 发送请求。
 * 
 * @return 0 成功，-1 失败(超时)，-2 失败(已断开)。
*/
int abcdk_comm_easy_request(abcdk_comm_easy_t *easy, abcdk_comm_message_t *req,
                            abcdk_comm_message_t **rsp, time_t timeout);

/**
 * 启动监听。
 * 
 * @warning 当通信对象不再需要时，
 *
 * @return !NULL(0) 成功(对象指针)，NULL(0) 失败。
*/
abcdk_comm_easy_t *abcdk_comm_easy_listen(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr,
                                          abcdk_comm_easy_request_cb request_cb, void *opaque);

/**
 * 启动连接。
 *
 * @return !NULL(0) 成功(对象指针)，NULL(0) 失败。
*/
abcdk_comm_easy_t *abcdk_comm_easy_connect(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr,
                                           abcdk_comm_easy_request_cb request_cb, void *opaque);

__END_DECLS

#endif //ABCDK_COMM_EASY_H