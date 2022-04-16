/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_SIGNAL_H
#define ABCDK_UTIL_SIGNAL_H

#include "util/general.h"
#include "util/thread.h"

__BEGIN_DECLS

/**
 * 简单的信号处理。
*/
typedef struct _abcdk_signal
{
    /** 
     * 信号集合。
    */
    sigset_t signals;
    
    /**
     * 处理函数，NULL(0)忽略信号。
     * 
     * @return -1 终止，1 继续。
    */
    int (*signal_cb)(const siginfo_t *info, void *opaque);

    /**
     * 环境指针。
    */
    void *opaque;
}abcdk_signal_t;

/**
 * 等待信号。
 * 
 * @param timeout 超时(毫秒)。 >= 0 有信号或时间过期，< 0 直到有信号或出错。
*/
void abcdk_sigwaitinfo(abcdk_signal_t *sig, time_t timeout);

/**
 * 等待信号(异步)。
*/
void abcdk_sigwaitinfo_async(abcdk_signal_t *sig);

__END_DECLS

#endif //ABCDK_UTIL_SIGNAL_H