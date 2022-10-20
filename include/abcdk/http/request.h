/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_HTTP_REQUEST_H
#define ABCDK_HTTP_REQUEST_H

#include "abcdk/util/general.h"
#include "abcdk/util/mmap.h"
#include "abcdk/util/http.h"
#include "abcdk/comm/message.h"

__BEGIN_DECLS

/** HTTP请求对象。*/
typedef struct _abcdk_http_request abcdk_http_request_t;

/**
 * 减少对象的引用计数。
 * 
 * @warning 当引用计数为0时，对像将被删除。
*/
void abcdk_http_request_unref(abcdk_http_request_t **req);

/**
 * 增加对象的引用计数。
*/
abcdk_http_request_t *abcdk_http_request_refer(abcdk_http_request_t *src);

/**
 * 创建请求对象。
 * 
 * @param [in] max_size 最大长度(头部+实体)。
 * @param [in] buffer_point 缓存目录(实体有效)。NULL(0) 不启用。
*/
abcdk_http_request_t *abcdk_http_request_alloc(size_t max_size,const char *buffer_point);

/**
 * 获取实体。
 * 
 * @param [in] off 偏移量。
 * 
 * @return !NULL(0) 实体的指针，NULL(0) 无请实体。
*/
const void *abcdk_http_request_body(abcdk_http_request_t *req, off_t off);

/**
 * 获取实体长度。
*/
size_t abcdk_http_request_body_length(abcdk_http_request_t *req);

/**
 * 获取头部环境参数。
 * 
 * @param [in] line 行号，从0开始。
 * 
 * @return !NULL(0) 参数的指针，NULL(0) 超出头部范围。
*/
const char *abcdk_http_request_env(abcdk_http_request_t *req,int line);

/**
 * 匹配头部环境参数，返回变量的值。。
 * 
 * @return !NULL(0) 参数值的指针，NULL(0) 超出头部范围。
*/
const char *abcdk_http_request_getenv(abcdk_http_request_t *req, const char *name);

/**
 * 附加消息。
 * 
 * @param [in out] remain 缓存剩余的数据长度，返回时填充。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_http_request_append(abcdk_http_request_t *req,const void *data,size_t size,size_t *remain);

__END_DECLS

#endif //ABCDK_HTTP_REQUEST_H