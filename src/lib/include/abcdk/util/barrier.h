/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2026 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_BARRIER_H
#define ABCDK_UTIL_BARRIER_H

#include "abcdk/util/defs.h"

__BEGIN_DECLS

/**
 * 屏障.
*/
typedef struct _abcdk_barrier abcdk_barrier_t;

/** 销毁.*/
void abcdk_barrier_destroy(abcdk_barrier_t **ctx);

/** 创建.*/
abcdk_barrier_t *abcdk_barrier_create(size_t count);

/**
 * 等待.
 * 
 * @return 0 成功; < 0 失败.
*/
int abcdk_barrier_wait(abcdk_barrier_t *ctx);


__END_DECLS

#endif // ABCDK_UTIL_BARRIER_H
