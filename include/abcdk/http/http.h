/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_HTTP_HTTP_H
#define ABCDK_HTTP_HTTP_H

#include "abcdk/http/reply.h"
#include "abcdk/http/request.h"

__BEGIN_DECLS

/** 
 * HTTP回调函数。
 * 
 * @note 服务端新的连接会复制成员指针。
*/
typedef struct _abcdk_http_callback
{
    /**
     * 为新连接做准备工作的通知回调函数。
     * 
     * @note 如果未指定，则创建默认节点。
     * 
     * @param [out] node 新的节点，返回时填写。
     * 
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
     * 请求数据到达通知回调函数。
     * 
     * @param [out] next_proto 下层协议。0 HTTP/RTSP/RTP，1 隧道。
    */
    void (*request_cb)(abcdk_comm_node_t *node, abcdk_http_request_t *req,int *next_proto);

    /** 
     * 连接关闭通知回调函数。
     * 
     * @note 如果未指定，直接关闭。
    */
    void (*close_cb)(abcdk_comm_node_t *node);

}abcdk_http_callback_t;

/**
 * 申请通讯对象。
 *
 * @param [in] ctx 通讯环境指针。
 * @param [in] userdata 用户数据长度。
 * @param [in] max 最大长度(封包或单次)。
 * @param [in] tempdir 缓存目录。NULL(0) 忽略。
 *
 * @return !NULL(0) 成功(通讯对象指针)，NULL(0) 失败。
 */
abcdk_comm_node_t *abcdk_http_alloc(abcdk_comm_t *ctx, size_t userdata, size_t max, const char *tempdir);

/**
 * 监听客户端连接。
 * 
 * @param [in] node 通讯对象指针。
 * @param [in] ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param [in] addr 监听地址指针。
 * @param [in] cb 回调函数指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr,abcdk_http_callback_t *cb);


__END_DECLS

#endif //ABCDK_HTTP_HTTP_H