/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_PROC_H
#define ABCDK_SHELL_PROC_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"
#include "abcdk/util/signal.h"
#include "abcdk/util/exec.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/popen.h"
#include "abcdk/log/logger.h"


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
 * 单实例模式运行。
 * 
 * 文件句柄在退出前不要关闭，否则会使文件解除锁定状态。
 * 
 * 进程ID以十进制文本格式写入文件，例：2021 。
 * 
 * @param pid 正在运行的进程ID，当接口返回时填写。NULL(0) 忽略。
 * 
 * @return >= 0 成功(文件句柄，当前进程是唯一进程)，-1 失败(已有实例正在运行)。
*/
int abcdk_proc_singleton(const char* lockfile,int* pid);

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
 * 守护进程。
 * 
 * @param [in] interval 重启间隔(秒)。
 * @param [in] process_cb 子进程入口函数。
 * @param [in] opaque 子进程环境指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_proc_daemon(int interval, abcdk_fork_process_cb process_cb, void *opaque);

/**
 * 守护进程。
 * 
 * @param [in] interval 重启间隔(秒)。
 * @param [in] cmdline 命令行。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_proc_daemon2(int interval, const char *cmdline);



__END_DECLS

#endif //ABCDK_SHELL_PROC_H