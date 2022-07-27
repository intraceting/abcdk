/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_EASY_H
#define ABCDK_COMM_EASY_H

#include "comm/comm.h"
#include "comm/message.h"
#include "comm/queue.h"
#include "comm/waiter.h"

__BEGIN_DECLS

/** 简单的通信对象。*/
typedef struct _abcdk_comm_easy abcdk_comm_easy_t;

/** 
 * 请求回调函数。
 * 
 * @param easy 应用层环境指针。
 * @param req 请求数据指针。NULL(0) 连接或监听关闭。
 * @param len 请求数据长度。0 连接或监听关闭。
*/
typedef void (*abcdk_comm_easy_request_cb)(abcdk_comm_easy_t *easy, const void *req, size_t len);

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
 * @warning 1、看门狗精度为1000毫秒；2、超时生效时间受引擎的工作周期影响。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_easy_set_timeout(abcdk_comm_easy_t *easy, time_t timeout);

/**
 * 获取地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_easy_get_sockaddr(abcdk_comm_easy_t *easy, abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote);

/**
 * 获取地址(转换成字符串)。
 * 
 * @note unix/IPv4/IPv6有效。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_easy_get_sockaddr_str(abcdk_comm_easy_t *easy, char local[NAME_MAX],char remote[NAME_MAX]);

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
 * 调整私有数据空间大小。
 * 
 * @return !NULL(0) 成功(私有数据指针)，NULL(0) 失败。
*/
void *abcdk_comm_easy_private_resize(abcdk_comm_easy_t *easy, size_t size);

/**
 * 获取私有数据空间指针。
 * 
 * @return !NULL(0) 成功(私有数据指针)，NULL(0) 失败。
*/
void *abcdk_comm_easy_private_data(abcdk_comm_easy_t *easy);

/**
 * 获取私有数据空间大小。
*/
size_t abcdk_comm_easy_private_size(abcdk_comm_easy_t *easy);

/** 
 * 发送请求。
 * 
 * @param data 请求数据的指针。
 * @param len 请求数据的长度。
 * @param rsp 应答容器的指针，NULL(0) 不需要应答。
 * 
 * @return 0 成功，-1 失败(未发送/无应答)，-2 失败(已断开)。
*/
int abcdk_comm_easy_request(abcdk_comm_easy_t *easy, const void *data, size_t len,
                            abcdk_comm_message_t **rsp);

/** 
 * 发送应答。
 * 
 * @warning 仅限在请求回调函数中使用。
 * 
 * @param data 应答数据的指针。
 * @param len 应答数据的长度。
 * 
 * @return 0 成功，-1 失败(其它)，-2 失败(已断开)。
*/
int abcdk_comm_easy_response(abcdk_comm_easy_t *easy, const void *data, size_t len);

/**
 * 启动监听。
 * 
 * @param ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param addr 监听地址指针。
 * @param event_cb 事件回调函数指针(新的连接会复制这个指针)。
 * @param opaque 监听环境指针(新的连接会复制这个指针)。
 * 
 * @return !NULL(0) 成功(对象指针)，NULL(0) 失败。
*/
abcdk_comm_easy_t *abcdk_comm_easy_listen(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr,
                                          abcdk_comm_easy_request_cb request_cb, void *opaque);

/**
 * 启动连接。
 * 
 * @param ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param addr 服务端地址指针。
 * @param event_cb 事件回调函数指针。
 * @param opaque 客户端环境指针。
 * 
 * @return !NULL(0) 成功(对象指针)，NULL(0) 失败。
*/
abcdk_comm_easy_t *abcdk_comm_easy_connect(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr,
                                           abcdk_comm_easy_request_cb request_cb, void *opaque);

__END_DECLS

#endif //ABCDK_COMM_EASY_H