/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_RECEIVER_H
#define ABCDK_RECEIVER_H

#include "abcdk/util/path.h"
#include "abcdk/util/mmap.h"
#include "abcdk/util/endian.h"
#include "abcdk/util/bit.h"

__BEGIN_DECLS

/** 接收器对象。*/
typedef struct _abcdk_receiver abcdk_receiver_t;

/** 接收器协议。*/
typedef enum _abcdk_receiver_protocol
{
    /**Stream */
    ABCDK_RECEIVER_PROTO_STREAM = 0,
#define ABCDK_RECEIVER_PROTO_STREAM ABCDK_RECEIVER_PROTO_STREAM

    /**HTTP(0.9,1.0,1.1) RTSP(1.0)*/
    ABCDK_RECEIVER_PROTO_HTTP = 1,
#define ABCDK_RECEIVER_PROTO_HTTP ABCDK_RECEIVER_PROTO_HTTP
#define ABCDK_RECEIVER_PROTO_RTSP ABCDK_RECEIVER_PROTO_HTTP

    /**HTTP-Chunked(0.9,1.0,1.1) */
    ABCDK_RECEIVER_PROTO_CHUNKED = 2,
#define ABCDK_RECEIVER_PROTO_CHUNKED ABCDK_RECEIVER_PROTO_CHUNKED

    /**RTCP */
    ABCDK_RECEIVER_PROTO_RTCP = 3,
#define ABCDK_RECEIVER_PROTO_RTCP ABCDK_RECEIVER_PROTO_RTCP

    /**SMB */
    ABCDK_RECEIVER_PROTO_SMB = 4,
#define ABCDK_RECEIVER_PROTO_SMB ABCDK_RECEIVER_PROTO_SMB
#define ABCDK_RECEIVER_PROTO_CIFS ABCDK_RECEIVER_PROTO_SMB

    /**SMB-Half */
    ABCDK_RECEIVER_PROTO_SMB_HALF = 5
#define ABCDK_RECEIVER_PROTO_SMB_HALF ABCDK_RECEIVER_PROTO_SMB_HALF
    
}abcdk_receiver_protocol_t;

/**
 * 减少引用计数。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_receiver_unref(abcdk_receiver_t **ctx);

/**
 * 增加引用计数。
*/
abcdk_receiver_t *abcdk_receiver_refer(abcdk_receiver_t *src);

/**
 * 创建对象。
 * 
 * @param [in] tempdir 缓存目录。NULL(0) 忽略。
*/
abcdk_receiver_t *abcdk_receiver_alloc(int protocol, size_t max, const char *tempdir);

/**
 * 附加消息。
 * 
 * @param [out] remain 缓存剩余的数据长度。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_receiver_append(abcdk_receiver_t *ctx,const void *data,size_t size,size_t *remain);

/**
 * 获取数据。
 * 
 * @param [in] off 偏移量。
 * 
 * @return !NULL(0) 数据的指针，NULL(0) 无数据。
*/
const void *abcdk_receiver_data(abcdk_receiver_t *ctx, off_t off);

/**
 * 获取数据长度。
*/
size_t abcdk_receiver_length(abcdk_receiver_t *ctx);

/**
 * 获取头部长度。
*/
size_t abcdk_receiver_header_length(abcdk_receiver_t *ctx);

/**
 * 获取实体长度。
*/
size_t abcdk_receiver_body_length(abcdk_receiver_t *ctx);

/**
 * 获取实体。
 * 
 * @param [in] off 偏移量。
 * 
 * @return !NULL(0) 实体的指针，NULL(0) 无实体。
*/
const void *abcdk_receiver_body(abcdk_receiver_t *ctx, off_t off);


/**
 * 获取头部环境参数。
 * 
 * @param [in] line 行号，从0开始。
 * 
 * @return !NULL(0) 参数的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_receiver_header_line(abcdk_receiver_t *ctx,int line);

/**
 * 查找头部环境参数的值。
 * 
 * @return !NULL(0) 参数值的指针，NULL(0) 超出头部范围或无头部信息。
*/
const char *abcdk_receiver_header_line_getenv(abcdk_receiver_t *ctx, const char *name, uint8_t delim);

__END_DECLS

#endif //ABCDK_RECEIVER_RECEIVER_H
