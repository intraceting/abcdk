/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_SPINLOCK_H
#define ABCDK_UTIL_SPINLOCK_H

#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"

__BEGIN_DECLS

/**
 * 自旋锁。
*/
typedef struct _abcdk_spinlock abcdk_spinlock_t;

/** 销毁。*/
void abcdk_spinlock_destroy(abcdk_spinlock_t **ctx);

/** 创建。*/
abcdk_spinlock_t *abcdk_spinlock_create();


/**
 * 加锁。
 * 
 * @param block !0 直到成功或出错返回，0 尝试一下即返回。
 * 
 * @return 0 成功，!0 出错。
 * 
*/
int abcdk_spinlock_lock(abcdk_spinlock_t *ctx, int block);

/**
 * 解锁。
 * 
 * @return 0 成功；!0 出错。
*/
int abcdk_spinlock_unlock(abcdk_spinlock_t *ctx);


__END_DECLS

#endif // ABCDK_UTIL_SPINLOCK_H
