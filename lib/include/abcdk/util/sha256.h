/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_SHA256_H
#define ABCDK_UTIL_SHA256_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/mmap.h"

__BEGIN_DECLS

/** 简单的SHA256。*/
typedef struct _abcdk_sha256 abcdk_sha256_t;

/**销毁。*/
void abcdk_sha256_destroy(abcdk_sha256_t **ctx);

/*创建。*/
abcdk_sha256_t *abcdk_sha256_create();

/*重置 。*/
void abcdk_sha256_reset(abcdk_sha256_t *ctx);

/**更新。*/
void abcdk_sha256_update(abcdk_sha256_t *ctx, const void *data, size_t size);

/**结束。*/
void abcdk_sha256_final(abcdk_sha256_t *ctx, uint8_t hashcode[32]);

/**结束。*/
void abcdk_sha256_final2hex(abcdk_sha256_t *ctx,char hashcode[65],int ABC);

/**计算内存块的SHA256。*/
int abcdk_sha256_once(const void *data, size_t size, uint8_t hashcode[32]);

/**
 * 计算内存块的SHA256，并转换成字符串。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sha256_from_buffer2string(const void *data,size_t size,char hashcode[65],int ABC);

/**
 * 计算文件的SHA256，并转换成字符串。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sha256_from_file2string(const char *file,char hashcode[65],int ABC);

__END_DECLS

#endif //ABCDK_UTIL_SHA256_H