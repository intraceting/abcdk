/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_MUTEX_H
#define ABCDK_UTIL_MUTEX_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

/**
 * 互斥量、事件。
 * 
 * @note 如果需要支持跨进程特性，需要让结构体数据运行在共享内存中。
*/
typedef struct _abcdk_mutex
{
    /**
     * 事件属性。
    */
    pthread_condattr_t condattr;

    /**
     * 事件。
    */
    pthread_cond_t cond;

    /**
     * 互斥量属性。
    */
    pthread_mutexattr_t mutexattr;

    /**
     * 互斥量。
    */
    pthread_mutex_t mutex;

} abcdk_mutex_t;


/** 销毁。*/
void abcdk_mutex_destroy(abcdk_mutex_t *ctx);

/** 初始化。*/
void abcdk_mutex_init(abcdk_mutex_t *ctx);

/**
 * 初始化。
 * 
 * @note 当互斥量拥用共享属性时，在多进程间有效。
 * 
 * @param shared 0 私有，!0 共享。
*/
void abcdk_mutex_init2(abcdk_mutex_t *ctx, int shared);

/**
 * 加锁。
 * 
 * @param block !0 直到成功或出错返回，0 尝试一下即返回。
 * 
 * @return 0 成功，!0 出错。
 * 
*/
int abcdk_mutex_lock(abcdk_mutex_t *ctx, int block);

/**
 * 解锁。
 * 
 * @return 0 成功；!0 出错。
*/
int abcdk_mutex_unlock(abcdk_mutex_t *ctx);

/**
 * 等待事件通知。
 * 
 * @param timeout 超时(毫秒)。< 0 直到有事件或出错。
 * 
 * @return 0 成功(有事件)；!0 超时或出错(errno)。
*/
int abcdk_mutex_wait(abcdk_mutex_t *ctx, time_t timeout);

/**
 * 发出事件通知。
 * 
 * @param broadcast 是否广播事件通知。0 否，!0 是。
 * 
 * @return 0 成功；!0 出错。
*/
int abcdk_mutex_signal(abcdk_mutex_t *ctx, int broadcast);


__END_DECLS

#endif // ABCDK_UTIL_MUTEX_H
