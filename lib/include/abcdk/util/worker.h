/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_UTIL_WORKER_H
#define ABCDK_UTIL_WORKER_H

#include "abcdk/util/general.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/queue.h"
#include "abcdk/util/wred.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS

/**简单的线程池。*/
typedef struct _abcdk_worker abcdk_worker_t;

/**线程池的配置。*/
typedef struct _abcdk_worker_config
{
    /*线程数量。默认：CPU核心数量。*/
    int numbers;

    /**
     * WRED最小阈值。 
     * 
     * @note 有效范围：200~6000，默认：800
    */
    int wred_min_th;

    /**
     * WRED最大阈值。 
     * 
     * @note 有效范围：400~8000，默认：1000
    */
    int wred_max_th;

    /**
     * WRED权重因子。
     * 
     * @note 有效范围：1~99，默认：2 
    */
    int wred_weight;

    /**
     * WRED概率因子。 
     * 
     * @note 有效范围：1~99，默认：2 
    */
    int wred_prob;

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

/**
 * 派发。
 *
 * @param [in] key 关键的。!0 是，0 否。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_worker_dispatch_ex(abcdk_worker_t *ctx,uint64_t event,void *item,int key);

__END_DECLS

#endif // ABCDK_UTIL_WORKER_H