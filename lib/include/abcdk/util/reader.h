/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_READER_H
#define ABCDK_UTIL_READER_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/** 读者环境。*/
typedef struct _abcdk_reader abcdk_reader_t;

/** 销毁。*/
void abcdk_reader_destroy(abcdk_reader_t **reader);

/** 
 * 创建。
 * 
 * @param blksize 块大小(字节)。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_reader_t *abcdk_reader_create(size_t blksize);

/** 停止。*/
void abcdk_reader_stop(abcdk_reader_t *reader);

/** 
 * 启动。
 * 
 * @param fd 文件句柄。
 * 
 * @return 0 成功，-1 失败，-2 失败(已经启动)。
*/
int abcdk_reader_start(abcdk_reader_t *reader,int fd);

/** 
 * 读。
 * 
 * @return > 0 成功。= 0 已经到末尾，< 0 失败。
*/
ssize_t abcdk_reader_read(abcdk_reader_t *reader,void *buf,size_t size);

__END_DECLS

#endif //ABCDK_UTIL_READER_H