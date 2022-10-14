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

__BEGIN_DECLS

/**
 * 创建子进程。
 * 
 * @param [in] cmd 可执行程序(包括路径)。
 * @param [in] args 参数。
 * @param [in] envs 环境变量。
 * @param [in] uid 用户ID，0 忽略。
 * @param [in] gid 用户组ID，0 忽略。
 * @param [in] rpath 根路径，NULL(0) 忽略。
 * @param [in] wpath 工作路径，NULL(0) 忽略。
 *  
 * @return 0 成功，!0 失败。
 *
*/
int abcdk_exec(const char *cmd, char *const *args, char *const *envs,
               uid_t uid, gid_t gid, const char *rpath, const char *wpath);

__END_DECLS

#endif //ABCDK_UTIL_EXEC_H