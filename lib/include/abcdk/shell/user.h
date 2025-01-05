/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_SHELL_USER_H
#define ABCDK_SHELL_USER_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"

__BEGIN_DECLS

/**
 * 获取当前用户的运行路径。
 * 
 * @note 可能不存在，使用前最好检查一下。
 * @note /var/run/user/$UID/
 * 
 * @param append 拼接目录或文件名。NULL(0) 忽略。
*/
char* abcdk_user_dirname(char* buf,const char* append);

__END_DECLS

#endif //ABCDK_SHELL_USER_H