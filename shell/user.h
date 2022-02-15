/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_USER_H
#define ABCDK_SHELL_USER_H

#include "util/general.h"

__BEGIN_DECLS

/**
 * 获取当前用户的运行路径。
 * 
 * 可能不存在，使用前最好检查一下。
 *
 * /var/run/user/$UID/
 * 
 * @param append 拼接目录或文件名。NULL(0) 忽略。
*/
char* abcdk_user_dirname(char* buf,const char* append);

__END_DECLS

#endif //ABCDK_SHELL_USER_H