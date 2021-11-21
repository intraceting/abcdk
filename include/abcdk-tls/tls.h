/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_TLS_TLS_H
#define ABCDK_TLS_TLS_H

#include "abcdk-util/general.h"
#include "abcdk-util/getargs.h"
#include "abcdk-util/socket.h"
#include "abcdk-util/epollex.h"
#include "abcdk-util/openssl.h"

__BEGIN_DECLS

/**/
#ifndef HEADER_SSL_H
typedef struct ssl_ctx_st SSL_CTX;
#endif //HEADER_SSL_H

/**/
typedef struct _abcdk_tls_node *abcdk_tls_node;

/* TLS事件。*/
enum _abcdk_tls_event
{
    /*已连接。*/
    ABCDK_TLS_EVENT_CONNECT = 1,
#define ABCDK_TLS_EVENT_CONNECT ABCDK_TLS_EVENT_CONNECT

    /*有数据到达。*/
    ABCDK_TLS_EVENT_INPUT = 2,
#define ABCDK_TLS_EVENT_INPUT ABCDK_TLS_EVENT_INPUT

    /*链路空闲，可以发送。*/
    ABCDK_TLS_EVENT_OUTPUT = 3,
#define ABCDK_TLS_EVENT_OUTPUT ABCDK_TLS_EVENT_OUTPUT

    /*已断开。*/
    ABCDK_TLS_EVENT_CLOSE = 4
#define ABCDK_TLS_EVENT_CLOSE ABCDK_TLS_EVENT_CLOSE
};

/*事件回调函数。*/
typedef void (*abcdk_tls_event_cb)(abcdk_tls_node *node, uint32_t event, void *opaque);

/**
 * 设置超时。
 * 
 * @param timeout 超时(毫秒)
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_set_timeout(abcdk_tls_node *node, time_t timeout);

/**
 * 获取远端地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_get_peername(abcdk_tls_node *node, abcdk_sockaddr_t *addr);

/**
 * 读。
 * 
 * @return > 0 已读取数据的长度，0 正在关闭，-1 无数据。
*/
ssize_t abcdk_tls_read(abcdk_tls_node *node, void *buf, size_t size);

/**
 * 监听是否可读。
 * 
 * @note 当读权利被占用时，不会有其它线程获得读事件。
 * 
 * @param done 0 仅监听，!0 释放读权利(非权利拥有者无效)。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_read_watch(abcdk_tls_node *node,int done);

/**
 * 写。
 * 
 * @return > 0 已写入数据的长度，0 正在关闭，-1 链路忙。
*/
ssize_t abcdk_tls_write(abcdk_tls_node *node, void *buf, size_t size);

/**
 * 监听是否可写。
 * 
 * @note 当写权利被占用时，不会有其它线程获得写事件。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_write_watch(abcdk_tls_node *node);

/**
 * 消息循环。
*/
void abcdk_tls_loop(abcdk_tls_event_cb event_cb);

/**
 * 监听客户端连接。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_listen(abcdk_sockaddr_t *addr, SSL_CTX *ssl_ctx, void *opaque);

/**
 * 连接远程服务器。
 * 
 * @warning 仅发出连接指令，连接是否成功以消息通知。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_connect(abcdk_sockaddr_t *addr, SSL_CTX *ssl_ctx, void *opaque);

__END_DECLS

#endif //ABCDK_TLS_TLS_H