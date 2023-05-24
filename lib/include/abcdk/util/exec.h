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
 * 执行新程序，替换当前进程。
 * 
 * @param [in] filename 可执行程序或脚本文件(包括路径)。
 * @param [in] args 参数。
 * @param [in] envs 环境变量，NULL(0) 继承当前进程。
 * @param [in] uid 用户ID，0 忽略。
 * @param [in] gid 用户组ID，0 忽略。
 * @param [in] rpath 根路径，NULL(0) 忽略。
 * @param [in] wpath 工作路径，NULL(0) 忽略。
 *  
 * @return 0 成功，!0 失败。
 *
*/
int abcdk_exec(const char *filename, char *const *args, char *const *envs,
               uid_t uid, gid_t gid, const char *rpath, const char *wpath);

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