/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_MD5_H
#define ABCDK_UTIL_MD5_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS

/** 简单的MD5。*/
typedef struct _abcdk_md5 abcdk_md5_t;

/** 销毁。*/
void abcdk_md5_destroy(abcdk_md5_t **ctx);

/** 创建。*/
abcdk_md5_t *abcdk_md5_create();

/** 重置。*/
void abcdk_md5_reset(abcdk_md5_t *ctx);

/** 更新。*/
void abcdk_md5_update(abcdk_md5_t *ctx, const void *data, size_t size);

/** 结束。*/
void abcdk_md5_final(abcdk_md5_t *ctx,uint8_t hashcode[16]);

/** 结束。*/
void abcdk_md5_final2hex(abcdk_md5_t *ctx,char hashcode[33],int ABC);

__END_DECLS

#endif //ABCDK_UTIL_MD5_H
