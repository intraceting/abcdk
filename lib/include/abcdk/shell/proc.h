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
int abcdk_proc_wait_exit_signal(abcdk_logger_t *logger, time_t timeout);

/**
 * 守护进程。
 * 
 * @param [in] interval 重启间隔(秒)。
 * @param [in] process_cb 子进程入口函数。
 * @param [in] opaque 子进程环境指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_proc_daemon(abcdk_logger_t *logger, int interval, abcdk_exec_fork_process_cb process_cb, void *opaque);

__END_DECLS

#endif //ABCDK_SHELL_PROC_H