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
#define ABCDK_TLS_EVENT_CONNECT     1
#define ABCDK_TLS_EVENT_INPUT       2
#define ABCDK_TLS_EVENT_OUTPUT      3
#define ABCDK_TLS_EVENT_CLOSE       4

/**
 * 设置超时。
 * 
 * @param timeout 超时(毫秒)
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_set_timeout(uint64_t tls, time_t timeout);

/**
 * 获取远端地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_get_peername(uint64_t tls, abcdk_sockaddr_t *addr);

/**
 * 读。
 * 
 * @return >0 成功(已读取数据的长度)，<=0 失败(无数据或正在关闭)。
*/
ssize_t abcdk_tls_read(uint64_t tls, void *buf, size_t size);

/**
 * 监听数据到达。
 * 
 * @warning 每次监听只会通知一次。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_read_watch(uint64_t tls);

/**
 * 写。
 * 
 * 
 * @return >0 成功(已写入数据的长度)，<=0 失败(未写入或正在关闭)。
*/
ssize_t abcdk_tls_write(uint64_t tls, void *buf, size_t size);

/**
 * 监听链路空闲。
 * 
 * @warning 每次监听只会通知一次。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tls_write_watch(uint64_t tls);

/**
 * 消息循环。
*/
void abcdk_tls_loop(void (*event_cb)(uint64_t tls, uint32_t event, void *opaque));

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