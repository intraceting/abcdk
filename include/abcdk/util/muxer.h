/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_MUXER_H
#define ABCDK_UTIL_MUXER_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/** 多路复用器对象。*/
typedef struct _abcdk_muxer abcdk_muxer_t;

/**
 * 多路复用器选项。
 */
typedef enum _abcdk_muxer_option
{
    /** 指令间隔(毫秒，64bits).*/
    ABCDK_MUXER_OPT_INTERVAL = 1,
#define ABCDK_MUXER_OPT_INTERVAL ABCDK_MUXER_OPT_INTERVAL

} abcdk_muxer_option_t;

/** 销毁对象。*/
void abcdk_muxer_destroy(abcdk_muxer_t **com);

/**
 * 创建对象。
 *
 * @return !NULL(0) 成功(对象指针)，NULL(0) 失败。
 */
abcdk_muxer_t *abcdk_muxer_create();

/** 加锁。*/
void abcdk_muxer_lock(abcdk_muxer_t *ctx);

/** 解锁。*/
void abcdk_muxer_unlock(abcdk_muxer_t *ctx);

/**
 * 绑定句柄。
 *
 * @warning 会强制给绑定名柄添加异步标志。
 *
 * @return 旧的句柄。
 */
int abcdk_muxer_attach(abcdk_muxer_t *ctx, int fd);

/**
 * 分离句柄。
 *
 * @return 旧的句柄。
 */
int abcdk_muxer_detach(abcdk_muxer_t *ctx);

/**
 * 设置选项。
 *
 * @param 0 成功，-1 失败。
 */
int abcdk_muxer_set_option(abcdk_muxer_t *ctx, int opt, ...);

/**
 * 获取选项。
 *
 * @param 0 成功，-1 失败。
 */
int abcdk_muxer_get_option(abcdk_muxer_t *ctx, int opt, ...);

/**
 * 传输数据。
 *
 * @param [in] out 输出数据，NULL(0) 忽略。
 * @param [in] outlen 输出数据长度，<= 0 忽略。
 * @param [out] in 输入数据，NULL(0) 忽略。
 * @param [in] inlen 输入数据长度，<= 0 忽略。
 * @param [in] timeout 超时(毫秒)。
 * @param [in] magic 起始码，NULL(0) 忽略。注：仅对输入有效。
 * @param [in] mglen 起始码长度，<= 0 忽略。注：仅对输入有效。
 *
 * @param 0 成功，-1 失败(或超时)。
 */
int abcdk_muxer_transfer(abcdk_muxer_t *ctx, const void *out, size_t outlen, void *in, size_t inlen,
                         time_t timeout, const void *magic, size_t mglen);

__END_DECLS

#endif // ABCDK_UTIL_MUXER_H