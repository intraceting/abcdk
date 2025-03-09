/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_CLOCK_H
#define ABCDK_UTIL_CLOCK_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/time.h"

__BEGIN_DECLS

/**
 * 计时器(微秒)。
 * 
 * @param [in] start 起始时刻。
 * @param [out] dot 打点时刻，NULL(0) 忽略。
 * 
 * @return 两次打点间隔时长。
*/
uint64_t abcdk_clock(uint64_t start,uint64_t *dot);

/**
 * 计算剩余时长。
 *
 * @param [in] start 起始时刻。
 * @param [in] duration 有效时长。
 * @param [in] check 检查时刻。
 * 
 * @return > 0 剩余时长，-1 已过期或未生效。
 */
int64_t abcdk_clock_remainder(uint64_t start, uint64_t duration, uint64_t check);

__END_DECLS

#endif //ABCDK_UTIL_CLOCK_H