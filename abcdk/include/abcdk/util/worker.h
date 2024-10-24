/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_WORKER_H
#define ABCDK_UTIL_WORKER_H

#include "abcdk/util/general.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/queue.h"

__BEGIN_DECLS

/**简单的线程池。*/
typedef struct _abcdk_worker abcdk_worker_t;

/**线程池的配置。*/
typedef struct _abcdk_worker_config
{
    /*线程数量。<= 0 使用CPU核心数。*/
    int numbers;

    /**环境指针。*/
    void *opaque;

    /**处理回调函数。*/
    void (*process_cb)(void *opaque,uint64_t event,void *item);

} abcdk_worker_config_t;

/**
 * 停止。
 * 
 * @warning 新的项目将被拒绝，队列清空后返回。
*/
void abcdk_worker_stop(abcdk_worker_t **ctx);

/** 启动。*/
abcdk_worker_t *abcdk_worker_start(abcdk_worker_config_t *cfg);

/**
 * 派发。
 *
 * @return 0 成功，-1 失败。
*/
int abcdk_worker_dispatch(abcdk_worker_t *ctx,uint64_t event,void *item);

__END_DECLS

#endif // ABCDK_UTIL_WORKER_H