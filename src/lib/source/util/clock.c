/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/clock.h"


uint64_t abcdk_clock(uint64_t start,uint64_t *dot)
{
    uint64_t current = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 9);

    if(dot)
        *dot = current;
    
    return (current - start);
}

int64_t abcdk_clock_remainder(uint64_t start, uint64_t duration, uint64_t check)
{
    int64_t remainder = 0;

    /*开始时刻大于当前时刻，则未生效。*/
    if (start > check)
        return -1;

    /*结束时刻小于当前时刻，则已过期。*/
    if (start + duration < check)
        return -1;

    /*运行时长不能超过有效时长。*/
    if (duration <= (check - start))
        return -1;

    /*计算剩余时长。*/
    remainder = duration - (check - start);

    return remainder;
}