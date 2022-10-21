/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_PATH_H
#define ABCDK_UTIL_PATH_H

#include "abcdk/util/defs.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS

/**
 * 拼接目录。
 * 
 * @note 自动检查前后的'/'字符，接拼位置只保留一个'/'字符，或自动添加一个'/'字符。
 * 
 * @warning 要有足够的可用空间，不然会溢出。
*/
char *abcdk_dirdir(char *path,const char *suffix);

/**
 * 创建目录。
 * 
 * @note 支持创建多级目录。如果末尾不是'/'，则最后一级的名称会被当做文件名而忽略。
*/
void abcdk_mkdir(const char *path,mode_t mode);

/**
 * 截取目录。
 * 
 * @note 最后一级的名称会被裁剪，并且无论目录结构是否真存在都会截取。 
*/
char *abcdk_dirname(char *dst, const char *src);

/**
 * 截取目录或文件名称。
 * 
 * @note 最后一级的名称'/'(包括)之前的会被裁剪，并且无论目录结构是否真存在都会截取。 
*/
char *abcdk_basename(char *dst, const char *src);

/**
 * 美化目录。
 * 
 * @note 不会检测目录结构是否存在。
 * 
 * 例：/aaaa/bbbb/../ccc -> /aaaa/ccc
 * 例：/aaaa/bbbb/./ccc -> /aaaa/bbbb/ccc
*/
char *abcdk_dirnice(char *dst, const char *src);

/**
 * 修补文件或目录的绝对路径。
 * 
 * @note 不会检测目录结构是否存在。
 * 
 * @param file 文件或目录的指针。
 * @param path 路径的指针，NULL(0) 当前工作路径。
*/
char *abcdk_abspath(char *buf, const char *file, const char *path);

__END_DECLS

#endif //ABCDK_UTIL_PATH_H