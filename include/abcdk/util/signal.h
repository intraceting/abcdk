/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_SIGNAL_H
#define ABCDK_UTIL_SIGNAL_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS

/**
 * 等待信号。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return > 0 有信号，<= 0 超时或出错。
*/
int abcdk_signal_wait(const sigset_t *sigs, siginfo_t *info, time_t timeout);

/**
 * 填充信号集合。
 * 
 * @param sigdel 排除的信号，-1 结束。
 */
void abcdk_signal_fill(sigset_t *sigs,int sigdel,...);

/**
 * 阻塞信号集合。
 * 
 * @param news 新的信号集合。NULL(0) 忽略。
 * @param olds 旧的信号集合。NULL(0) 忽略。

 * @return 0 成功，-1 失败。
 */
int abcdk_signal_block(const sigset_t *news,sigset_t *olds);


__END_DECLS

#endif //ABCDK_UTIL_SIGNAL_H