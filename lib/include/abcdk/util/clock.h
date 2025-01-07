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
 * @param [in] start 起始刻度。
 * @param [out] dot 打点刻度，NULL(0) 忽略。
 * 
 * @return 两次打点间隔时长。
*/
uint64_t abcdk_clock(uint64_t start,uint64_t *dot);

__END_DECLS

#endif //ABCDK_UTIL_CLOCK_H