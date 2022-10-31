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
   * @param [out] result 0 允许连接，-1 禁止连接。
   */
  void (*accept_cb)(abcdk_comm_node_t *node, int *result);

  /**
   * 请求回调函数。
   *
   * @param req 请求数据。
   */
  void (*request_cb)(abcdk_comm_node_t *node, abcdk_http_request_t *req);

  /**
   * 拉取通知回调函数。
   *
   * @warning 仅在发送队列从忙碌状态切换到空闲状态时，发出通知。
  */
  void (*fetch_cb)(abcdk_comm_node_t *node);

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
 * @return 0 成功，-1 失败。
 */
int abcdk_http_send(abcdk_comm_node_t *node, const void *data,size_t size);

/** 
 * 发送数据。
 * 
 * @param [in] max 格式化数据最大长度。 
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_send_vformat(abcdk_comm_node_t *node, int max, const char *fmt, va_list ap);

/** 
 * 发送数据。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_send_format(abcdk_comm_node_t *node, int max, const char *fmt, ...);

/**
 * 发送数据。
 * 
 * @warning 内存对象将被托管，应用层不可以继续访问内存对象。
 * 
 * @param [in] obj 内存对象指针，索引0号元素有效。注：仅做指针复制，不会改变对象的引用计数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_send_object(abcdk_comm_node_t *node, abcdk_object_t *obj);

/**
 * 启动监听。
 * 
 * @warning 新的连接会复制“用户环境指针”。
 * 
 * @param [in] ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param [in] addr 监听地址指针。
 * @param [in] cb 事件回调函数指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_http_callback_t *cb);

__END_DECLS

#endif //ABCDK_HTTP_HTTP_H