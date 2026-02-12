/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SYSTEM_USER_H
#define ABCDK_SYSTEM_USER_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"

__BEGIN_DECLS

/**
 * 获取当前用户的运行路径.
 * 
 * @note 可能不存在, 使用前最好检查一下.
 * @note /var/run/user/$UID/
 * 
 * @param append 拼接目录或文件名.NULL(0) 忽略.
*/
char* abcdk_user_dir_run(char* buf,const char* append);

/**
 * 获取当前用户的家路径.
 * 
 * @note 可能不存在, 使用前最好检查一下.
 * @note $HOME;
 * 
 * @param append 拼接目录或文件名.NULL(0) 忽略.
*/
char* abcdk_user_dir_home(char* buf,const char* append);

__END_DECLS

#endif //ABCDK_SYSTEM_USER_H