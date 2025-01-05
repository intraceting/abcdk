/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_RWLOCK_H
#define ABCDK_UTIL_RWLOCK_H

#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"

__BEGIN_DECLS

/**
 * 读写锁。
*/
typedef struct _abcdk_rwlock abcdk_rwlock_t;

/** 销毁。*/
void abcdk_rwlock_destroy(abcdk_rwlock_t **ctx);

/** 创建。*/
abcdk_rwlock_t *abcdk_rwlock_create();

/**
 * 读锁。
 * 
 * @param block !0 直到成功或出错返回，0 尝试一下即返回。
 * 
 * @return 0 成功，!0 出错。
 * 
*/
int abcdk_rwlock_rdlock(abcdk_rwlock_t *ctx, int block);

/**
 * 写锁。
 * 
 * @param block !0 直到成功或出错返回，0 尝试一下即返回。
 * 
 * @return 0 成功，!0 出错。
 * 
*/
int abcdk_rwlock_wrlock(abcdk_rwlock_t *ctx, int block);

/**
 * 解锁。
 * 
 * @return 0 成功；!0 出错。
*/
int abcdk_rwlock_unlock(abcdk_rwlock_t *ctx);


__END_DECLS

#endif // ABCDK_UTIL_RWLOCK_H
