/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_MUTEX_H
#define ABCDK_UTIL_MUTEX_H

#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"

__BEGIN_DECLS

/**
 * 互斥量、事件。
*/
typedef struct _abcdk_mutex abcdk_mutex_t;


/** 销毁。*/
void abcdk_mutex_destroy(abcdk_mutex_t **ctx);

/** 创建。*/
abcdk_mutex_t *abcdk_mutex_create();

/**
 * 加锁。
 * 
 * @param block !0 直到成功或出错返回，0 尝试一下即返回。
 * 
 * @return 0 成功，!0 出错。
 * 
*/
int abcdk_mutex_lock(abcdk_mutex_t *ctx, int block);

/**
 * 解锁。
 * 
 * @return 0 成功；!0 出错。
*/
int abcdk_mutex_unlock(abcdk_mutex_t *ctx);

/**
 * 等待事件通知。
 * 
 * @param timeout 时长(毫秒)。< 0 直到有事件或出错。
 * 
 * @return 0 成功(有事件)；!0 超时或出错(errno)。
*/
int abcdk_mutex_wait(abcdk_mutex_t *ctx, time_t timeout);

/**
 * 发出事件通知。
 * 
 * @param broadcast 是否广播事件通知。0 否，!0 是。
 * 
 * @return 0 成功；!0 出错。
*/
int abcdk_mutex_signal(abcdk_mutex_t *ctx, int broadcast);


__END_DECLS

#endif // ABCDK_UTIL_MUTEX_H
