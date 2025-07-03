/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_MMAN_H
#define ABCDK_UTIL_MMAN_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/io.h"
#include "abcdk/util/shm.h"

__BEGIN_DECLS

/**
 * 刷新数据。
 * 
 * @warning 如果映射的内存页面是私有模式，对数据修改不会影响原文件。
 * 
 * @param async 0 同步，!0 异步。
 * 
 * @return 0 成功，-1 失败。
*/
ABCDK_DEPRECATED
int abcdk_msync(abcdk_object_t* obj,int async);

/**
 * 映射文件到内存页面。
 * 
 * @note 文件句柄可以提前关闭。
 * 
 * @param [in] truncate 截断文件(或扩展文件)。0 忽略。
 *
 * @return NULL(0) 失败，!NULL(0) 成功。
*/
ABCDK_DEPRECATED
abcdk_object_t* abcdk_mmap_fd(int fd,size_t truncate,int rw,int shared);

/**
 * 映射文件到内存页面。
 * 
 * @return NULL(0) 失败，!NULL(0) 成功。
*/
ABCDK_DEPRECATED
abcdk_object_t* abcdk_mmap_filename(const char* name,size_t truncate,int rw,int shared,int create);

/**
 * 映射临时文件到内存页面。
 * 
 * @return NULL(0) 失败，!NULL(0) 成功。
*/
ABCDK_DEPRECATED
abcdk_object_t* abcdk_mmap_tempfile(char* name,size_t truncate,int rw,int shared);

#if !defined(__ANDROID__)

/**
 * 映射共离内存文件到内存页面。
 * 
 * @return NULL(0) 失败，!NULL(0) 成功。
*/
ABCDK_DEPRECATED
abcdk_object_t* abcdk_mmap_shm(const char* name,size_t truncate,int rw,int shared,int create);

#endif //__ANDROID__

/**
 * 重新映射。
 * 
 * @return 0 成功，-1 失败。
*/
ABCDK_DEPRECATED
int abcdk_mremap(abcdk_object_t* obj,size_t truncate,int rw,int shared);

__END_DECLS


#endif //ABCDK_UTIL_MMAN_H
