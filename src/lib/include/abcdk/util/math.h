/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_MATH_H
#define ABCDK_UTIL_MATH_H

#include "abcdk/util/defs.h"

__BEGIN_DECLS

/**最小公倍数.*/
uint64_t abcdk_math_lcm(uint64_t a, uint64_t b);

/**数值归一化.*/
double abcdk_math_sigmoid(double x);

/**欧几里得归一化(特征向量).*/
void abcdk_math_normalize_l2(float *data, int len);

/**
 * 获取钟摆的位置.
 * 
 * @param t 时间
 * @param min 摆动幅度最小值
 * @param max 摆动幅度最大值
 * 
 */
uint64_t abcdk_math_ticker_pos(uint64_t t, uint64_t min, uint64_t max);

__END_DECLS

#endif //ABCDK_UTIL_MATH_H