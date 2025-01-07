/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_POOL_H
#define ABCDK_UTIL_POOL_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/thread.h"

__BEGIN_DECLS

/**
 * 一个简单的池子。
*/
typedef struct _abcdk_pool abcdk_pool_t;

/**
 * 销毁。
*/
void abcdk_pool_destroy(abcdk_pool_t **ctx);

/**
 * 创建。
 * 
 * @param size 大小(单元格)。
 * @param number 数量(单元格)。
*/
abcdk_pool_t *abcdk_pool_create(size_t size, size_t number);

/**
 * 拉取数据。
 * 
 * @return 0 成功，!0 失败(空了)。
*/
int abcdk_pool_pull(abcdk_pool_t *ctx, void *buf);

/**
 * 推送数据。
 * 
 * @return 0 成功，!0 失败(满了)。
*/
int abcdk_pool_push(abcdk_pool_t *ctx, const void *buf);

/**
 * 队列长度。
*/
size_t abcdk_pool_count(abcdk_pool_t *ctx);

__END_DECLS

#endif //ABCDK_UTIL_POOL_H