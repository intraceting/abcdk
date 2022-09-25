/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_HTTP_HTTP_H
#define ABCDK_HTTP_HTTP_H

#include "util/http.h"
#include "comm/comm.h"
#include "comm/message.h"
#include "comm/queue.h"
#include "http/request.h"

/**
 * 通讯对象的回调函数。
 *
 * @warning 服务端新的连接会复制成员指针。
 */
typedef struct _abcdk_http_callback
{
  /**
   * 新连接通知回调函数。
   *
   * @return 0 允许连接，-1 禁止连接。
   */
  void (*accept_cb)(abcdk_comm_node_t *node, int *result);

  /**
   * 请求回调函数。
   *
   * @param req 请求数据。
   */
  void (*request_cb)(abcdk_comm_node_t *node, abcdk_http_request_t *req);

  /** 连接关闭通知回调函数。*/
  void (*close_cb)(abcdk_comm_node_t *node);

} abcdk_http_callback_t;

/**
 * 申请通讯对象。
 *
 * @param [in] ctx 通讯环境指针。
 * @param [in] up_max_size 上行最大长度。
 *
 * @return !NULL(0) 成功(通讯对象指针)，NULL(0) 失败。
 */
abcdk_comm_node_t *abcdk_http_alloc(abcdk_comm_t *ctx,size_t up_max_size);

/** 
 * 发送应答。
 * 
 * @param data 应答数据的指针。
 * @param len 应答数据的长度。
 * 
 * @return 0 成功，-1 失败(其它)，-2 失败(已断开)。
*/
int abcdk_http_response(abcdk_comm_node_t *node, const void *data, size_t len);

/** 
 * 发送应答。
 * 
 * @warning 内存对象将被托管，应用层不可以继续访问内存对象。
 * 
 * @param [in] data 应答数据对象的指针。注：仅做指针复制，不会改变对象的引用计数。
 * 
 * @return 0 成功，-1 失败(其它)，-2 失败(已断开)。
*/
int abcdk_http_response2(abcdk_comm_node_t *node,abcdk_object_t *data);

/** 
 * 应答结束。
 * 
 * @warning 连接复用前需要调用此方法，否则应答数据发送后将会关闭连接。
 * 
 * @return 0 成功，-1 失败(其它)，-2 失败(已断开)。
*/
int abcdk_http_response_end(abcdk_comm_node_t *node);

/**
 * 启动监听。
 * 
 * @warning 新的连接会复制“用户环境指针”。
 * 
 * @param [in] ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param [in] addr 监听地址指针。
 * @param [in] cb 事件回调函数指针。
 * 
 * @return !NULL(0) 成功(对象指针)，NULL(0) 失败。
*/
int abcdk_http_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_http_callback_t *cb);

#endif //ABCDK_HTTP_HTTP_H