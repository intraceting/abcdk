/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_POPEN_H
#define ABCDK_UTIL_POPEN_H

#include "abcdk/util/exec.h"

__BEGIN_DECLS

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

#endif //ABCDK_UTIL_POPEN_H