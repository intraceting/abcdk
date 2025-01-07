/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_DIRENT_H
#define ABCDK_UTIL_DIRENT_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/path.h"
#include "abcdk/util/fnmatch.h"

__BEGIN_DECLS

/**
 * 打开目录。
 * 
 * @note 已打开目录进行压栈缓存。
 * 
 * @return 0 成功，-1 失败(不影响已经打开目录)。
*/
int abcdk_dirent_open(abcdk_tree_t **dir,const char *path);

/**
 * 读取目录。
 * 
 * @note 如果已经当前目录没有未读取的子项，则关闭当前目录，回退到上一个打开的目录。
 * 
 * @param [in] pattern 通配符。NULL(0) 忽略。
 * @param [in] fullpath 是否读取全路径。!0 是，0 否。
 * 
 * @return 0 成功，-1 失败(无子项)。
*/
int abcdk_dirent_read(abcdk_tree_t *dir,const char *pattern,char file[PATH_MAX],int fullpath);

__END_DECLS

#endif //ABCDK_UTIL_DIRENT_H