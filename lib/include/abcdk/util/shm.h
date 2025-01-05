/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_SHM_H
#define ABCDK_UTIL_SHM_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

#if !defined(__ANDROID__)

/**
 * 打开共享内存文件。
 *
 * @note 通常是在'/dev/shm/'目录内创建。
 * 
 * @return >= 0 句柄，-1 失败。
*/
int abcdk_shm_open(const char* name,int rw, int create);

/**
 * 删除共享内存文件。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_shm_unlink(const char* name);

#endif //__ANDROID__

__END_DECLS

#endif //ABCDK_UTIL_SHM_H
