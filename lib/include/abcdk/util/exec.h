/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_UTIL_EXEC_H
#define ABCDK_UTIL_EXEC_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/**
 * 子进程入口函数。
 *
 * @return 出错码。0~126之间有效。
 */
typedef int (*abcdk_fork_process_cb)(void *opaque);

/**
 * 创建子进程。
 *
 * @param [in] process_cb 子进程入口函数。
 * @param [in] opaque 环境指针。
 * @param [out] stdin_fd 输入句柄。NULL(0) 忽略。
 * @param [out] stdout_fd 输出句柄。NULL(0) 忽略。
 * @param [out] stderr_fd 出错句柄。NULL(0) 忽略。
 *
 * @return >= 0 成功(PID)，-1 失败。
 */
pid_t abcdk_fork(abcdk_fork_process_cb process_cb, void *opaque,
                 int *stdin_fd, int *stdout_fd, int *stderr_fd);

/**
 * 创建子进程，并执行SHELL命令。
 *
 * @param [in] filename 可执行程序或脚本(包括路径)。
 * @param [in] args 参数(二维数组，NULL(0)结束)。
 * @param [in] envs 环境变量(二维数组，NULL(0)结束)。NULL(0) 忽略，继承父进程。
 * @param [in] uid 用户ID。0 忽略，继承父进程。
 * @param [in] gid 用户组ID。0 忽略，继承父进程。
 * @param [in] rpath 根目录。NULL(0) 忽略，继承父进程。
 * @param [in] wpath 工作目录。NULL(0) 忽略，继承父进程。
 *
 * @return >= 0 成功(PID)，-1 失败。
 */
pid_t abcdk_system(const char *filename, char *const *args, char *const *envs,
                   uid_t uid, gid_t gid, const char *rpath, const char *wpath,
                   int *stdin_fd, int *stdout_fd, int *stderr_fd);

/**
 * 创建子进程，用于执行shell。
 *
 * @param cmdline 命令行。
 *
 * @return 子进程ID 成功，-1 失败。
 */
pid_t abcdk_popen(const char *cmdline, char *const *envs, uid_t uid,
                  gid_t gid, const char *rpath, const char *wpath,
                  int *stdin_fd, int *stdout_fd, int *stderr_fd);
                  
__END_DECLS

#endif // ABCDK_UTIL_EXEC_H