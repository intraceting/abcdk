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
 * 通讯对象的回调函数。
 * 
 * @warning 服务端新的连接会复制成员指针。
*/
typedef struct _abcdk_http_callback
{
    /**
     * 为新连接做准备工作的通知回调函数。
     * 
     * @warning 如果未指定，则创建默认节点。
     * 
     * @param [out] node 新的节点，返回时填写。
     * 
    */
    void (*prepare_cb)(abcdk_comm_node_t **node, abcdk_comm_node_t *listen);

    /**
     * 请求数据到达通知回调函数。
     * 
    */
    void (*request_cb)(abcdk_comm_node_t *node, abcdk_http_request_t *req);

}abcdk_http_callback_t;


__END_DECLS

#endif //ABCDK_HTTP_HTTP_H