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

double abcdk_math_sigmoid(double x)
{
    return ((double)(1.0 / (1.0 + exp(-x))));
}

void abcdk_math_normalize_l2(float *data, int len)
{
    float norm2 = 0.f;

    assert(data != NULL && len > 0);

    for (int i = 0; i < len; i++)
    {
        norm2 += data[i] * data[i];
    }

    float norm = sqrt(norm2);
    for (int i = 0; i < len; i++)
    {
        data[i] = data[i] / norm;
    }
}