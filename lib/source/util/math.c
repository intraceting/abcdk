/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/math.h"

uint64_t abcdk_math_lcm(uint64_t a, uint64_t b)
{
    uint64_t m;

    m = ABCDK_MAX(a, b);

    for (uint64_t i = m; i > 0; i++)
        if (i % a == 0 && i % b == 0)
            return i;

    return 0;
}