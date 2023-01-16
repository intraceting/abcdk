/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_HTTP_RECEIVER_H
#define ABCDK_HTTP_RECEIVER_H

#include "abcdk/util/general.h"
#include "abcdk/util/mmap.h"
#include "abcdk/util/endian.h"
#include "abcdk/util/http.h"
#include "abcdk/util/url.h"
#include "abcdk/util/receiver.h"

__BEGIN_DECLS

/** HTTP接收器对象。*/
typedef struct _abcdk_http_receiver abcdk_http_receiver_t;

/** 接收器协议。*/
typedef enum _abcdk_http_receiver_protocol
{
    /** HTTP RTSP*/
    ABCDK_HTTP_RECEIVER_PROTO_NATURAL = 0,
#define ABCDK_HTTP_RECEIVER_PROTO_NATURAL ABCDK_HTTP_RECEIVER_PROTO_NATURAL
#define ABCDK_HTTP_RECEIVER_PROTO_HTTP ABCDK_HTTP_RECEIVER_PROTO_NATURAL
#define ABCDK_HTTP_RECEIVER_PROTO_RTSP ABCDK_HTTP_RECEIVER_PROTO_NATURAL

    /** Chunked */
    ABCDK_HTTP_RECEIVER_PROTO_CHUNKED = 1,
#define ABCDK_HTTP_RECEIVER_PROTO_CHUNKED ABCDK_HTTP_RECEIVER_PROTO_CHUNKED

    /** RTCP */
    ABCDK_HTTP_RECEIVER_PROTO_RTCP = 2,
#define ABCDK_HTTP_RECEIVER_PROTO_RTCP ABCDK_HTTP_RECEIVER_PROTO_RTCP

    /** Tunnel */
    ABCDK_HTTP_RECEIVER_PROTO_TUNNEL = 3
#define ABCDK_HTTP_RECEIVER_PROTO_TUNNEL ABCDK_HTTP_RECEIVER_PROTO_TUNNEL

}abcdk_http_receiver_protocol_t;

/**
 * 减少对象的引用计数。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_http_receiver_unref(abcdk_http_receiver_t **rec);

/**
 * 增加对象的引用计数。
*/
abcdk_http_receiver_t *abcdk_http_receiver_refer(abcdk_http_receiver_t *src);

/**
 * 创建对象。
 * 
 * @param [in] proto 协议。
 * @param [in] max 最大长度。
 * @param [in] tempdir 缓存目录。NULL(0) 忽略。
*/
abcdk_http_receiver_t *abcdk_http_receiver_alloc(int proto, size_t max, const char *tempdir);

/**
 * 附加消息。
 * 
 * @param [out] remain 剩余的数据长度。NULL(0) 未知的流数据。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_http_receiver_append(abcdk_http_receiver_t *rec,const void *data,size_t size,size_t *remain);

/**
 * 获取实体。
 * 
 * @param [in] off 偏移量。
 * 
 * @return !NULL(0) 实体的指针，NULL(0) 无请实体。
*/
const void *abcdk_http_receiver_body(abcdk_http_receiver_t *rec, off_t off);

/**
 * 获取实体长度。
*/
size_t abcdk_http_receiver_body_length(abcdk_http_receiver_t *rec);

/**
 * 获取头部环境参数。
 * 
 * @param [in] line 行号，从0开始。
 * 
 * @return !NULL(0) 参数的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_receiver_header(abcdk_http_receiver_t *rec,int line);

/**
 * 查找头部环境参数的值。
 * 
 * @return !NULL(0) 参数值的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_http_receiver_getenv(abcdk_http_receiver_t *rec, const char *name);


__END_DECLS

#endif //ABCDK_HTTP_RECEIVER_H