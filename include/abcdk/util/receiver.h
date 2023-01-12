/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_RECEIVER_H
#define ABCDK_RECEIVER_H

#include "abcdk/util/path.h"
#include "abcdk/util/mmap.h"

__BEGIN_DECLS

/** 接收器对象。*/
typedef struct _abcdk_receiver abcdk_receiver_t;

/** 消息协议。*/
typedef struct _abcdk_receiver_protocol
{
    /** 环境指针。*/
    void *opaque;

    /**
     * 消息解包回调函数。
     * 
     * @param [out] diff 差额(待增量)。
     *
     * @return 1 消息包完整，0 需要更多数据，-1 不支持的协议(或有错发生)。
     */
    int (*unpack_cb)(void *opaque, abcdk_receiver_t *msg,size_t *diff);

} abcdk_receiver_protocol_t;

/**
 * 减少引用计数。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_receiver_unref(abcdk_receiver_t **msg);

/**
 * 增加引用计数。
*/
abcdk_receiver_t *abcdk_receiver_refer(abcdk_receiver_t *src);

/**
 * 创建对象。
 * 
 * @param [in] tempdir 缓存目录。NULL(0) 忽略。
*/
abcdk_receiver_t *abcdk_receiver_alloc(const char *tempdir);

/**
 * 获取指针。
*/
void *abcdk_receiver_data(const abcdk_receiver_t *msg);

/**
 * 获取长度。
*/
size_t abcdk_receiver_size(const abcdk_receiver_t *msg);

/**
 * 获取偏移量。
*/
size_t abcdk_receiver_offset(const abcdk_receiver_t *msg);

/**
 * 调整缓存大小。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_receiver_resize(abcdk_receiver_t *msg, size_t size);

/**
 * 设置数据包协议。
*/
void abcdk_receiver_protocol_set(abcdk_receiver_t *msg, abcdk_receiver_protocol_t *prot);

#define abcdk_receiver_protocol_set_simple(msg, op, cb) \
    {                                                  \
        abcdk_receiver_protocol_t prot = {0};           \
        prot.opaque = op;                              \
        prot.unpack_cb = cb;                           \
        abcdk_receiver_protocol_set(msg, &prot);        \
    }

/**
 * 附加消息。
 * 
 * @param [out] remain 缓存剩余的数据长度。
 * 
 * @return 1 缓存区已满，0 缓存区未满，-1 有错误发生。
*/
int abcdk_receiver_append(abcdk_receiver_t *msg,const void *data,size_t size,size_t *remain);


__END_DECLS

#endif //ABCDK_RECEIVER_RECEIVER_H
