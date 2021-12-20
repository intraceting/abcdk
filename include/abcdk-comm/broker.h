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
typedef void (*abcdk_broker_message_cb)(abcdk_broker_node_t *node,abcdk_comm_msg_t *msg,void *opaque);

/**
 * 节点引用释放。
*/
void abcdk_broker_node_unref(abcdk_broker_node_t **node);

/**
 * 节点增加引用。
 * 
 * @return !NULL(0) 成功(节点的指针)，NULL(0) 失败。
*/
abcdk_broker_node_t *abcdk_broker_node_refer(abcdk_broker_node_t *src);

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
 * 投递消息。
 * 
 * @warning 消息将被托管，应用层不可以继续访问被投递的消息对象。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_broker_post(abcdk_broker_node_t *node, abcdk_comm_msg_t *msg);

/**
 * 启动监听。
 * 
 * @param message_cb 消息回调函数指针。
 * @param opaque 应用层环境指针。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_broker_listen(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_broker_message_cb message_cb, void *opaque);

/**
 * 启动连接。
 * 
 * @warning 只要消息回调函数没有通知连接已经断开，则链路可用。
 * @warning 当通信节点指针不在需要时，需要应用层主动释放。
 * 
 * @param message_cb 消息回调函数指针。
 * @param opaque 应用层环境指针。
 * 
 * @return !NULL(0) 成功(通信节点指针)，NULL(0) 失败。
*/
abcdk_broker_node_t *abcdk_broker_connect(SSL_CTX *ssl_ctx,abcdk_sockaddr_t *addr, abcdk_broker_message_cb message_cb, void *opaque);


__END_DECLS

#endif //ABCDK_COMM_BROKER_H