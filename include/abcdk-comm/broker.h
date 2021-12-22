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

/** 事件回调函数。*/
typedef void (*abcdk_broker_event_cb)(abcdk_broker_node_t *node, uint32_t event);

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
 * @warning 1、看门狗精度为200毫秒；2、超时生效时间受引擎的工作周期影响。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_broker_set_timeout(abcdk_broker_node_t *node, time_t timeout);

/**
 * 获取本机地址。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_broker_get_sockname(abcdk_broker_node_t *node, abcdk_sockaddr_t *addr);

/**
 * 获取远端地址。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_broker_get_peername(abcdk_broker_node_t *node, abcdk_sockaddr_t *addr);

/**
 * 设置应用层环境指针。
 * 
 * @return 旧的指针。
*/
void *abcdk_broker_set_userdata(abcdk_broker_node_t *node, void *opaque);

/**
 * 获取应用层环境指针。
 * 
 * @return 旧的指针。
*/
void *abcdk_broker_get_userdata(abcdk_broker_node_t *node);

/**
 * 读。
 * 
 * @note 当读权利被占用时，不会有其它线程获得读事件。
 * 
 * @return > 0 已读取数据的长度，0 无数据。
*/
ssize_t abcdk_broker_read(abcdk_broker_node_t *node, void *buf, size_t size);

/**
 * 监听是否可读。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_broker_read_watch(abcdk_broker_node_t *node);

/**
 * 投递消息。
 * 
 * @warning 消息将被托管，应用层不可以继续访问被投递的消息对象。
 * 
 * @return 0 成功，-1 失败(已断开)。
*/
int abcdk_broker_post(abcdk_broker_node_t *node, abcdk_comm_msg_t *msg);

/**
 * 启动监听。
 * 
 * @param event_cb 事件回调函数指针。
 * @param opaque 应用层环境指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_broker_listen(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_broker_event_cb event_cb, void *opaque);

/**
 * 启动连接。
 * 
 * @param event_cb 事件回调函数指针。
 * @param opaque 应用层环境指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_broker_connect(SSL_CTX *ssl_ctx,abcdk_sockaddr_t *addr, abcdk_broker_event_cb event_cb, void *opaque);


__END_DECLS

#endif //ABCDK_COMM_BROKER_H