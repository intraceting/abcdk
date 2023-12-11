/*
 * This file is part of ABCDK.
 * 
 * MIT License
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
typedef int (*abcdk_exec_fork_process_cb)(void *opaque);

/**
 * 创建子进程。
 *
 * @param [in] process_cb 子进程入口函数。
 * @param [in] opaque 环境指针。
 *
 * @return 0 成功，-1 失败。
 */
pid_t abcdk_exec_fork(abcdk_exec_fork_process_cb process_cb, void *opaque,
                      int *stdin_fd, int *stdout_fd, int *stderr_fd);

/**
 * 创建子进程。
 * 
 * @param stdin_fd 输入句柄，NULL(0) 忽略。
 * @param stdout_fd 输出句柄，NULL(0) 忽略。
 * @param stderr_fd 出错句柄，NULL(0) 忽略。
 * 
 * @return 子进程ID 成功，-1 失败。
 */
pid_t abcdk_exec_new(const char *filename, char *const *args, char *const *envs,
                     uid_t uid, gid_t gid, const char *rpath, const char *wpath,
                     int *stdin_fd, int *stdout_fd, int *stderr_fd);

__END_DECLS

#endif //ABCDK_UTIL_EXEC_H