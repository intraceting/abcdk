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