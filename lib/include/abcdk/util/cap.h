/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_UTIL_CAP_H
#define ABCDK_UTIL_CAP_H

#include "abcdk/util/general.h"

#ifdef HAVE_LIBCAP
#include <sys/capability.h>
#endif 

__BEGIN_DECLS

#ifdef _SYS_CAPABILITY_H

/**
 * 获取进程的权限。
 * 
 * @return 1 有权限，0 无权限，-1 未知。
*/
int abcdk_cap_get_pid(pid_t pid,cap_value_t power, cap_flag_t flag);

/**
 * 设置进程的权限。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_cap_set_pid(pid_t pid,cap_value_t power, cap_flag_t flag,cap_flag_value_t cmd);

#endif //_SYS_CAPABILITY_H

__END_DECLS

#endif //ABCDK_UTIL_CAP_H