/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_TIMER_H
#define ABCDK_UTIL_TIMER_H

#include "abcdk/util/defs.h"
#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/mutex.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/time.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS

/**简单的定时器。*/
typedef struct _abcdk_timer abcdk_timer_t;

/**
 * 定时器执行回调函数。
 * 
 * @return 间隔(毫秒)。
*/
typedef uint64_t (*abcdk_timer_routine_cb)(void *opaque);

/**销毁。*/
void abcdk_timer_destroy(abcdk_timer_t **ctx);

/**创建。*/
abcdk_timer_t *abcdk_timer_create(abcdk_timer_routine_cb routine_cb, void *opaque);

__END_DECLS

#endif //ABCDK_UTIL_TIMER_H