/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_THREAD_H
#define ABCDK_UTIL_THREAD_H

#include "abcdk/util/general.h"
#include "abcdk/util/mutex.h"

__BEGIN_DECLS

/** 线程对象。*/
typedef struct _abcdk_thread_t
{
    /** 句柄。*/
    pthread_t handle;

    /** 返回值。*/
    void* result;

    /** 线程函数。*/
    void *(*routine)(void *opaque);

    /** 环境指针。*/
    void *opaque;

} abcdk_thread_t;

/**
 * 创建线程。
 * 
 * @param joinable 线程结束后回收资源的方式。0 系统自动回收，!0 由调用者回收。
 * 
 * @return 0 成功；!0 出错。
 * 
*/
int abcdk_thread_create(abcdk_thread_t *ctx,int joinable);

/**
 * 等待线程结束并回收资源。
 * 
 * @note 当线程不支持等待时，直接返回。
 * 
 * @return 0 成功；!0 出错。
*/
int abcdk_thread_join(abcdk_thread_t* ctx);

/**
 * 创建线程组。
 * 
 * @param [in] count 数量。
 * @param [in] ctxs 线程对象(数组)。
 * 
 * @return 0 成功(创建的数量)；!0 出错。
*/
int abcdk_thread_create_group(int count,abcdk_thread_t ctxs[],int joinable);

/** 
 * 设置线程亲源CPU。
 * 
 * @param [in] cpus CPUID数组。CPUID从0开始，-1 结束。
 * 
 * @return 0 成功；!0 出错。
*/
int abcdk_thread_setaffinity(pthread_t tid,int cpus[]);

/**
 * 设置线程亲源CPU。
 */
int abcdk_thread_setaffinity2(pthread_t tid,int cpu);

/**
 * 选举主线程。
 * 
 * @note 非主线程调用此函数不会影向数据变化。
 * 
 * @return 0 当前线程为主线程；!0 当前线程非主线程。
 */
int abcdk_thread_leader_vote(volatile pthread_t *tid);

/**
 * 测试主线程。
 * 
 * @note 任何线程调用此函数不会影向数据变化。
 * 
 * @return 0 当前线程为主线程；!0 当前线程非主线程。
 */
int abcdk_thread_leader_test(const volatile pthread_t *tid);

/**
 * 主线程退出。
 * 
 * @note 非主线程调用此函数不会影向数据变化。
 *
 * @return 0 当前线程为主线程；!0 当前线程非主线程。
 */
int abcdk_thread_leader_quit(volatile pthread_t *tid);

/** 设置线程名字。*/ 
void abcdk_thread_setname(pthread_t ptd, const char *fmt, ...);

/** 获取线程名字。*/ 
char *abcdk_thread_getname(pthread_t ptd,char name[16]);

__END_DECLS

#endif // ABCDK_UTIL_THREAD_H
