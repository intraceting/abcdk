/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_HTTP_HTTP_H
#define ABCDK_HTTP_HTTP_H

#include "abcdk/util/general.h"
#include "abcdk/util/http.h"
#include "abcdk/comm/comm.h"
#include "abcdk/comm/message.h"
#include "abcdk/comm/queue.h"
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
 * @param [in] up_buffer_point 上行缓存目录(实体有效)。NULL(0) 不启用。
 *
 * @return !NULL(0) 成功(通讯对象指针)，NULL(0) 失败。
 */
abcdk_comm_node_t *abcdk_http_alloc(abcdk_comm_t *ctx,size_t up_max_size,const char *up_buffer_point);

/** 
 * 发送数据。
 * 
 * @warning 数据对象将被托管，应用层不可以继续访问内存对象。
 * 
 * @param [in] msg 数据对象的指针。注：仅做指针复制，不会改变对象的引用计数。
 * 
 * @return 0 成功，-1 失败(其它)，-2 失败(已断开)。
*/
int abcdk_http_send(abcdk_comm_node_t *node, abcdk_comm_message_t *msg);

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

__END_DECLS

#endif //ABCDK_HTTP_HTTP_H