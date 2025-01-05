/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
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
 * 设置信号集合。
 * 
 * @param op 操作码，0 添加，!0 删除。
 * @param sig 信号，-1 结束。
 */
void abcdk_signal_set(sigset_t *sigs,int op, int sig,...);

/**
 * 填充全部信号集合。
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

/**
 * 等待信号。
 * 
 * @param sigs 信号集合。NULL(0) 全部信号（SIGKILL和SIGSTOP除外）。
 * @param timeout 超时(毫秒)。
 * 
 * @return > 0 有信号，0 超时，-1 出错。
*/
int abcdk_signal_wait(siginfo_t *info, const sigset_t *sigs, time_t timeout);


__END_DECLS

#endif //ABCDK_UTIL_SIGNAL_H