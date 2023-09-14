/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_PARALLEL_H
#define ABCDK_UTIL_PARALLEL_H

#include "abcdk/util/general.h"
#include "abcdk/util/queue.h"

__BEGIN_DECLS

/** 并行计算环境。*/
typedef struct _abcdk_parallel abcdk_parallel_t;

/** 释放。*/
void abcdk_parallel_free(abcdk_parallel_t **ctx);

/** 创建。*/
abcdk_parallel_t *abcdk_parallel_alloc(size_t max);

/**
 * 实例回调函数。
 * 
 * @param [in] tid 线程编号，从0开始。
*/
typedef void (*abcdk_parallel_routine_cb)(void *opaque, uint32_t tid);

/**
 * 执行。
 * 
 * @param [in] number 线程数量。注：最终执行的数量，非并发数量。
 *
 * @return 0 成功，-1 失败。
 *
 */
int abcdk_parallel_run(abcdk_parallel_t *ctx,uint32_t number, void *opaque, abcdk_parallel_routine_cb routine_cb);

__END_DECLS

#endif // ABCDK_UTIL_PARALLEL_H