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
#include "abcdk/util/endian.h"
#include "abcdk/util/http.h"
#include "abcdk/util/url.h"
#include "abcdk/util/receiver.h"

__BEGIN_DECLS

/** HTTP请求对象。*/
typedef struct _abcdk_http_request abcdk_http_request_t;

/** 请求协议。*/
typedef enum _abcdk_http_request_protocol
{
    /** HTTP RTSP*/
    ABCDK_HTTP_REQUEST_PROTO_NATURAL = 0,
#define ABCDK_HTTP_REQUEST_PROTO_NATURAL ABCDK_HTTP_REQUEST_PROTO_NATURAL
#define ABCDK_HTTP_REQUEST_PROTO_HTTP ABCDK_HTTP_REQUEST_PROTO_NATURAL
#define ABCDK_HTTP_REQUEST_PROTO_RTSP ABCDK_HTTP_REQUEST_PROTO_NATURAL

    /** RTCP */
    ABCDK_HTTP_REQUEST_PROTO_RTCP = 1,
#define ABCDK_HTTP_REQUEST_PROTO_RTCP ABCDK_HTTP_REQUEST_PROTO_RTCP

    /** Tunnel */
    ABCDK_HTTP_REQUEST_PROTO_TUNNEL = 2
#define ABCDK_HTTP_REQUEST_PROTO_TUNNEL ABCDK_HTTP_REQUEST_PROTO_TUNNEL

}abcdk_http_request_protocol_t;

/**
 * 减少对象的引用计数。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_http_request_unref(abcdk_http_request_t **req);

/**
 * 增加对象的引用计数。
*/
abcdk_http_request_t *abcdk_http_request_refer(abcdk_http_request_t *src);

/**
 * 创建请求对象。
 * 
 * @param [in] proto 协议。
 * @param [in] max 最大长度。
 * @param [in] tempdir 缓存目录。NULL(0) 忽略。
*/
abcdk_http_request_t *abcdk_http_request_alloc(int proto, size_t max, const char *tempdir);

/**
 * 附加消息。
 * 
 * @param [out] remain 剩余的数据长度。NULL(0) 未知的流数据。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_http_request_append(abcdk_http_request_t *req,const void *data,size_t size,size_t *remain);

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
 * @return !NULL(0) 参数的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_request_env(abcdk_http_request_t *req,int line);

/**
 * 查找头部环境参数的值。
 * 
 * @return !NULL(0) 参数值的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_request_getenv(abcdk_http_request_t *req, const char *name);

/**
 * 获取请求方法。
 * 
 * @return !NULL(0) 请求方法的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_request_method(abcdk_http_request_t *req);

/**
 * 获取定位符。
 * 
 * @return !NULL(0) 定位符的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_request_location(abcdk_http_request_t *req);

/**
 * 获取协议和版本。
 * 
 * @return !NULL(0) 协议和版本的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_request_version(abcdk_http_request_t *req);

/**
 * 获取路径。
 * 
 * @return !NULL(0) 路径的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_request_path(abcdk_http_request_t *req);

/**
 * 获取路径参数。
 * 
 * @return !NULL(0) 路径参数的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_request_params(abcdk_http_request_t *req);

__END_DECLS

#endif //ABCDK_HTTP_REQUEST_H