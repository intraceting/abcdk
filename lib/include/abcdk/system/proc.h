/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SYSTEM_PROC_H
#define ABCDK_SYSTEM_PROC_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"
#include "abcdk/util/signal.h"
#include "abcdk/util/exec.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/popen.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS

/**
 * 获取当前程序的完整路径和文件名。
*/
char* abcdk_proc_pathfile(char* buf);

/**
 * 获取当前程序的完整路径。
 * 
 * @param append 拼接目录或文件名。NULL(0) 忽略。
 * 
*/
char* abcdk_proc_dirname(char* buf,const char* append);

/**
 * 获取当前程序的文件名。
*/
char* abcdk_proc_basename(char* buf);

/**
 * 单实例进程锁定。
 *
 * @note PID文件句柄在退出前不要关闭，否则会使文件解除锁定状态。
 *
 * @param [in] pid_fd PID文件句柄。
 * @param [out] pid 正在运行的进程ID。NULL(0) 忽略。
 *
 * @return 0 成功(当前进程是唯一进程)，-1 失败(已有实例正在运行)。
 */
int abcdk_proc_singleton_lock(int pid_fd, int *pid);

/**
 * 向单例进程发送信号。
 * 
 * @return 0 成功，-1 失败(不存在或已退出)。
*/
int abcdk_proc_singleton_kill(int pid_fd ,int signum);

/**
 * 执行外部命令。
*/
pid_t abcdk_proc_popen(int *stdin_fd, int *stdout_fd, int *stderr_fd, const char *cmd, ...);

/**
 * 执行外部命令。
*/
pid_t abcdk_proc_vpopen(int *stdin_fd, int *stdout_fd, int *stderr_fd, const char *cmd, va_list ap);

/**
 * 执行外部命令。
 * 
 * @return 0 成功(执行完成)，-1 失败(未找到或权限不足)。
*/
int abcdk_proc_shell(int *exitcode , int *sigcode,const char *cmd,...);

/**
 * 执行外部命令。
 * 
 * @return 0 成功(执行完成)，-1 失败(未找到或权限不足)。
*/
int abcdk_proc_vshell(int *exitcode , int *sigcode,const char *cmd,va_list ap);

/**
 * 拦截信号。
 * 
 * @param news 新的信号集合。NULL(0) 默认所有信号，SIGTRAP、SIGKILL、SIGSEGV、SIGSTOP除外。
 * @param olds 旧的信号集合。NULL(0) 忽略。

 * @return 0 成功，-1 失败。
*/
int abcdk_proc_signal_block(const sigset_t *news,sigset_t *olds);

/**
 * 等待信号。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return > 0 有信号，0 超时，-1 出错。
*/
int abcdk_proc_signal_wait(siginfo_t *info, time_t timeout);

/**
 * 等待终止信号。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return 1 收到终止信号，0 超时，-1 系统错误。
*/
int abcdk_proc_wait_exit_signal(time_t timeout);

/**
 * 运行子进程。
 * 
 * @param [in] process_cb 子进程入口函数。
 * @param [in] opaque 子进程环境指针。
 * @param [out] exitcode 状态码。
 * @param [out] sigcode 信号。
 * 
 * @return 0 成功(正常结束)，-1 失败(系统错误)，-2 失败(已结束或被终止)。
*/
int abcdk_proc_subprocess(abcdk_fork_process_cb process_cb, void *opaque,int *exitcode, int *sigcode);

/**
 * 运行子进程。
 * 
 * @param [in] cmdline 命令行。
 * @param [out] exitcode 状态码。
 * @param [out] sigcode 信号。
 * 
 * @return 0 成功(正常结束)，-1 失败(系统错误)，-2 失败(已结束或被终止)。
*/
int abcdk_proc_subprocess2(const char *cmdline,int *exitcode, int *sigcode);

__END_DECLS

#endif //ABCDK_SYSTEM_PROC_H