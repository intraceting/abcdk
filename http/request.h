/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_HTTP_REQUEST_H
#define ABCDK_HTTP_REQUEST_H

#include "util/general.h"
#include "util/mmap.h"
#include "util/http.h"
#include "comm/message.h"

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
 * @param [in] up_max_size 上行最大长度。
 * @param [in] buffer_point 缓存目录(实体有效)。NULL(0) 不启用。
*/
abcdk_http_request_t *abcdk_http_request_alloc(size_t up_max_size,const char *buffer_point);

/**
 * 获取实体。
 * 
 * @return !NULL(0) 实体的指针，NULL(0) 无请实体。
*/
const void *abcdk_http_request_body(abcdk_http_request_t *req);

/**
 * 获取头部环境参数。
 * 
 * @param [in] line 行号，从0开始。
 * 
 * @return !NULL(0) 参数的指针，NULL(0) 超出头部范围。
*/
const char *abcdk_http_request_env(abcdk_http_request_t *req,int line);

/**
 * 附加消息。
 * 
 * @param [in out] remain 缓存剩余的数据长度，返回时填充。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_http_request_append(abcdk_http_request_t *req,const void *data,size_t size,size_t *remain);

#endif //ABCDK_HTTP_REQUEST_H