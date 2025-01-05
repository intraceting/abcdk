/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_UTIL_STREAM_H
#define ABCDK_UTIL_STREAM_H

#include "abcdk/util/mutex.h"
#include "abcdk/util/queue.h"

__BEGIN_DECLS


/**简单的数据流。*/
typedef struct _abcdk_stream abcdk_stream_t;

/**
 * 销毁。
 * 
 * @note 与abcdk_stream_unref()作用相同。
*/
ABCDK_DEPRECATED
void abcdk_stream_destroy(abcdk_stream_t **ctx);

/**
 * 减少引用计数。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_stream_unref(abcdk_stream_t **ctx);

/**
 * 增加引用计数。
*/
abcdk_stream_t *abcdk_stream_refer(abcdk_stream_t *src);

/**创建。*/
abcdk_stream_t *abcdk_stream_create();

/**获取读或写的数量。*/
size_t abcdk_stream_size_tell(abcdk_stream_t *ctx,int writer);

/**
 * 读。
 * 
 * @note 非阻塞。
*/
ssize_t abcdk_stream_read(abcdk_stream_t *ctx,void *buf,size_t len);

/**
 * 写。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_stream_write_buffer(abcdk_stream_t *ctx,const void *buf,size_t len);

/**
 * 写。
 * 
 * @note 数据对象写入成功后将被托管，用户不可以再进行读和写操作。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_stream_write(abcdk_stream_t *ctx,abcdk_object_t *data);

__END_DECLS

#endif //ABCDK_UTIL_STREAM_H
