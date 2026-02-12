/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_PATH_H
#define ABCDK_UTIL_PATH_H

#include "abcdk/util/defs.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/string.h"
#include "abcdk/util/tree.h"

__BEGIN_DECLS

/**
 * 拼接路径.
 * 
 * @note 要有足够的可用空间, 不然会溢出.
*/
char *abcdk_dirdir(char *path,const char *suffix);

/**
 * 创建目录.
 * 
 * @note 支持创建多级目录.如果末尾不是'/', 则最后一级的名称会被当做文件名而忽略.
*/
void abcdk_mkdir(const char *path,mode_t mode);

/**
 * 截取路径.
 * 
 * @note 最后一级的名称会被裁剪, 并且无论目录结构是否真存在都会截取. 
*/
char *abcdk_dirname(char *dst, const char *src);

/**
 * 截取目录或文件名称.
 * 
 * @note 最后一级的名称'/'(包括)之前的会被裁剪, 并且无论目录结构是否真存在都会截取. 
*/
char *abcdk_basename(char *dst, const char *src);

/**
 * 去掉路径中冗余的信息.
 *
 * @note 不会检测目录结构是否存在.
 *
 * @param [in] decrease 缩减的深度.
 */
char *abcdk_abspath(char *buf, size_t decrease);



__END_DECLS

#endif //ABCDK_UTIL_PATH_H