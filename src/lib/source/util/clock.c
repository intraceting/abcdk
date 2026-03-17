/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/time.h"
#include "abcdk/util/clock.h"


uint64_t abcdk_clock(uint64_t start,uint64_t *dot)
{
    uint64_t current = abcdk_time_systime(9);

    if(dot)
        *dot = current;
    
    return (current - start);
}

int64_t abcdk_clock_remainder(uint64_t start, uint64_t duration, uint64_t check)
{
    int64_t remainder = 0;

    /*开始时刻大于当前时刻, 则未生效.*/
    if (start > check)
        return -1;

    /*结束时刻小于当前时刻, 则已过期.*/
    if (start + duration < check)
        return -1;

    /*运行时长不能超过有效时长.*/
    if (duration <= (check - start))
        return -1;

    /*计算剩余时长.*/
    remainder = duration - (check - start);

    return remainder;
}

void abcdk_clock_delay(uint64_t *dot, uint64_t duration)
{
    // 1. 获取当前时间, 但不立即更新 *dot
    uint64_t now = abcdk_time_systime(9);

    // 2. 计算从上次打点到现在的步长
    uint64_t step = (now > *dot) ? (now - *dot) : 0;

    // 3. 如果没到时间, 就睡够剩下的部分
    if (duration > step)
    {
        abcdk_nanosleep(duration - step);

        // 4. 关键：睡醒后, 将 dot 更新为“预期的理想时间点”
        // 这样可以彻底消除由于系统调度或 printf 耗时带来的微小偏差
        *dot += duration;
    }
    else
    {
        // 如果逻辑执行已经超时了, 则将 dot 同步为当前最新时间, 重新开始计时
        *dot = now;
    }
}