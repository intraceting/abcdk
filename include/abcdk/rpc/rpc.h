/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_RPC_RPC_H
#define ABCDK_RPC_RPC_H

#include "abcdk/util/comm.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/waiter.h"
#include "abcdk/util/bloom.h"

__BEGIN_DECLS

/**
 * RPC回调函数。
 *
 * @note 服务端新的连接会复制成员指针。
 */
typedef struct _abcdk_rpc_callback
{
    /** 
     * 为新连接做准备工作的通知回调函数。
     * 
     * @note 如果未指定，则创建默认节点。
     * 
     * @param [out] node 新的节点，返回时填写。
    */
    void (*prepare_cb)(abcdk_comm_node_t **node, abcdk_comm_node_t *listen);

    /**
     * 新连接通知回调函数。
     * 
     * @note 如果未指定，则接受所有连接。
     *
     * @param [out] result 0 允许连接，-1 禁止连接。
     */
    void (*accept_cb)(abcdk_comm_node_t *node, int *result);

    /**
     * 请求回调函数。
     *
     * @param node 通讯对象。
     * @param mid 消息ID。
     * @param req 消息指针。
     * @param len 消息长度。
     */
    void (*request_cb)(abcdk_comm_node_t *node, uint64_t mid, const void *req, size_t len);

    /** 
     * 连接关闭通知回调函数。
     * 
     * @note 如果未指定，直接关闭。
    */
    void (*close_cb)(abcdk_comm_node_t *node);

} abcdk_rpc_callback_t;

/**
 * 申请通讯对象。
 * 
 * @note 通讯协议相同才能相互收发消息。
 *
 * @param [in] ctx 通讯环境指针。
 * @param [in] protocol 通讯协议。
 * @param [in] userdata 用户数据长度。
 *
 * @return !NULL(0) 成功(通讯对象指针)，NULL(0) 失败。
 */
abcdk_comm_node_t *abcdk_rpc_alloc(abcdk_comm_t *ctx,uint32_t protocol,size_t userdata);

/**
 * 发送请求。
 *
 * @param data 请求数据的指针。
 * @param len 请求数据的长度。
 * @param rsp 应答容器的指针，NULL(0) 不需要应答。
 * @param timeout 应答等待时间(秒)。不需要应答时，此项将被忽略。
 *
 * @return 0 成功，-1 失败(未发送/无应答)，-2 失败(超时/已断开)。
 */
int abcdk_rpc_request(abcdk_comm_node_t *node, const void *data, size_t len,abcdk_object_t **rsp, time_t timeout);

/** 
 * 发送应答。
 * 
 * @param mid 消息ID。
 * @param data 应答数据的指针。
 * @param len 应答数据的长度。
 * 
 * @return 0 成功，-1 失败(其它)，-2 失败(已断开)。
*/
int abcdk_rpc_response(abcdk_comm_node_t *node, uint64_t mid, const void *data, size_t len);

/**
 * 启动监听。
 * 
 * @note 新的连接会复制“用户环境指针”。
 * 
 * @param [in] ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param [in] addr 监听地址指针。
 * @param [in] cb 通知回调函数指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_rpc_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_rpc_callback_t *cb);

/**
 * 启动连接。
 * 
 * @note 仅发出连接指令，连接是否成功以消息通知。
 * 
 * @param [in] ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param [in] addr 服务端地址指针。
 * @param [in] cb 通知回调函数指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_rpc_connect(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_rpc_callback_t *cb);

__END_DECLS

#endif //ABCDK_RPC_RPC_H