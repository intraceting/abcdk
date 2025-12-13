/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_UTIL_MD5_H
#define ABCDK_UTIL_MD5_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/mmap.h"

__BEGIN_DECLS

/** 简单的MD5.*/
typedef struct _abcdk_md5 abcdk_md5_t;

/** 销毁.*/
void abcdk_md5_destroy(abcdk_md5_t **ctx);

/** 创建.*/
abcdk_md5_t *abcdk_md5_create();

/** 重置.*/
void abcdk_md5_reset(abcdk_md5_t *ctx);

/** 更新.*/
void abcdk_md5_update(abcdk_md5_t *ctx, const void *data, size_t size);

/** 结束.*/
void abcdk_md5_final(abcdk_md5_t *ctx,uint8_t hashcode[16]);

/** 结束.*/
void abcdk_md5_final2hex(abcdk_md5_t *ctx,char hashcode[33],int ABC);

/**计算内存块的MD5.*/
int abcdk_md5_once(const void *data, size_t size, uint8_t hashcode[16]);

/**
 * 计算内存块的MD5, 并转换成字符串.
 * 
 * @return 0 成功, -1 失败.
*/
int abcdk_md5_from_buffer(const void *data,size_t size,char hashcode[33],int ABC);
#define abcdk_md5_from_buffer2string abcdk_md5_from_buffer

/**
 * 计算文件的MD5, 并转换成字符串.
 * 
 * @return 0 成功, -1 失败.
*/
int abcdk_md5_from_file(const char *file,char hashcode[33],int ABC);
#define abcdk_md5_from_file2string abcdk_md5_from_file

__END_DECLS

#endif //ABCDK_UTIL_MD5_H
