/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_HTTP_REPLY_H
#define ABCDK_HTTP_REPLY_H

#include "abcdk/comm/comm.h"

__BEGIN_DECLS

/**
 * 分块应答。
 * 
 * @warning 内存对象将被托管，应用层不可以继续访问内存对象。
 * 
 * @param [in] data 内存对象指针，索引0号元素有效。注：仅做指针复制，不会改变对象的引用计数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_reply_chunked(abcdk_comm_node_t *node, abcdk_object_t *data);

/**
 * 分块应答。
 * 
 * @param  [in] data 数据，NULL(0) 应答结束块。
 * @param  [in] size 长度，<= 0 应答结束块。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_reply_chunked_buffer(abcdk_comm_node_t *node, const void *data, size_t size);

/** 
 * 分块应答。
 * 
 * @param [in] max 格式化数据最大长度。 
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_reply_chunked_vformat(abcdk_comm_node_t *node, int max, const char *fmt, va_list ap);

/** 
 * 分块应答。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_http_reply_chunked_format(abcdk_comm_node_t *node, int max, const char *fmt, ...);

__END_DECLS

#endif //ABCDK_HTTP_REPLY_H