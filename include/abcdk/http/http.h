/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_HTTP_HTTP_H
#define ABCDK_HTTP_HTTP_H

#include "abcdk/util/comm.h"
#include "abcdk/http/receiver.h"

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
     *  输入数据到达通知回调函数。
     * 
     * @param [out] next_proto 下层协议。
    */
    void (*input_cb)(abcdk_comm_node_t *node, abcdk_http_receiver_t *rec,int *next_proto);

    /** 
     * 连接关闭通知回调函数。
     * 
     * @note 如果未指定，直接关闭。
    */
    void (*close_cb)(abcdk_comm_node_t *node);

    /** 
     * 输出队列空闲通知回调函数。
     * 
     * @note 如果未指定，则忽略。
    */
    void (*output_cb)(abcdk_comm_node_t *node);
    
    /** 
     * 连接完成通知回调函数。
     * 
     * @note 如果未指定，则忽略。
    */
    void (*connected_cb)(abcdk_comm_node_t *node,int *next_proto);

}abcdk_http_callback_t;

/**
 * 申请通讯对象。
 *
 * @param [in] ctx 通讯环境指针。
 * @param [in] userdata 用户数据长度。
 * @param [in] input_max 输入消息最大长度(封包或单次)。
 * @param [in] input_tempdir 输入消息缓存目录。NULL(0) 忽略。
 *
 * @return !NULL(0) 成功(通讯对象指针)，NULL(0) 失败。
 */
abcdk_comm_node_t *abcdk_http_alloc(abcdk_comm_t *ctx, size_t userdata, size_t input_max, const char *input_tempdir);

/**
 * 分块应答。
 * 
 * @note 内存对象将被托管，应用层不可以继续访问内存对象。
 * 
 * @param [in] data 内存对象指针，索引0号元素有效。注：仅做指针复制，不会改变对象的引用计数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_post_chunked(abcdk_comm_node_t *node, abcdk_object_t *data);

/**
 * 分块应答。
 * 
 * @param  [in] data 数据，NULL(0) 应答结束块。
 * @param  [in] size 长度，<= 0 应答结束块。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_post_chunked_buffer(abcdk_comm_node_t *node, const void *data, size_t size);

/** 
 * 分块应答。
 * 
 * @param [in] max 格式化数据最大长度。 
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_post_chunked_vformat(abcdk_comm_node_t *node, int max, const char *fmt, va_list ap);

/** 
 * 分块应答。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_post_chunked_format(abcdk_comm_node_t *node, int max, const char *fmt, ...);

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
int abcdk_http_connect(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_http_callback_t *cb);


__END_DECLS

#endif //ABCDK_HTTP_HTTP_H