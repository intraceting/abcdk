/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_BROKER_H
#define ABCDK_COMM_BROKER_H

#include "abcdk-comm/comm.h"
#include "abcdk-comm/message.h"

__BEGIN_DECLS

/** 通信节点。*/
typedef struct _abcdk_broker_node abcdk_broker_node_t;

/*消息回调函数。*/
typedef void (*abcdk_broker_message_cb)(abcdk_broker_node_t *node,const abcdk_comm_msg_t *req, abcdk_comm_msg_t **rsp,void *opaque);

/**
 * 设置超时。
 * 
 * @warning 1、看门狗精度为5000毫秒；2、超时生效时间受引擎的工作周期影响。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_broker_set_timeout(abcdk_broker_node_t *node, time_t timeout);

/**
 * 获取本机地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_broker_get_sockname(abcdk_broker_node_t *node, abcdk_sockaddr_t *addr);

/**
 * 获取远端地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_broker_get_peername(abcdk_broker_node_t *node, abcdk_sockaddr_t *addr);

/**
 * 启动监听。
 * 
 * @return 0 成功， !0 失败。
*/
int abcdk_broker_listen(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_broker_message_cb message_cb, void *opaque);

/**
 *  
*/
int abcdk_broker_transmit(abcdk_broker_node_t *node, abcdk_comm_msg_t *req, abcdk_comm_msg_t **rsp, time_t timeout);

__END_DECLS

#endif //ABCDK_COMM_BROKER_H