/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_STREAM_H
#define ABCDK_UTIL_STREAM_H

#include "abcdk/util/mutex.h"
#include "abcdk/util/queue.h"

__BEGIN_DECLS


/**简单的数据流。*/
typedef struct _abcdk_stream abcdk_stream_t;

/**销毁。*/
void abcdk_stream_destroy(abcdk_stream_t **ctx);

/**创建。*/
abcdk_stream_t *abcdk_stream_create();

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
