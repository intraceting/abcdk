/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/tenutil.h"

size_t abcdk_torch_tenutil_stride(int format, size_t width, size_t depth, size_t cell, size_t align)
{
    size_t stride = 0;

    assert(format == ABCDK_TORCH_TENFMT_NCHW || format == ABCDK_TORCH_TENFMT_NHWC);
    assert(width > 0 && depth > 0 && cell > 0);

    if (format == ABCDK_TORCH_TENFMT_NCHW)
    {
        stride = abcdk_align(width * cell, align);
    }
    else if (format == ABCDK_TORCH_TENFMT_NHWC)
    {
        stride = abcdk_align(width * depth * cell, align);
    }

    return stride;
}

size_t abcdk_torch_tenutil_size(int format, size_t block, size_t width, size_t stride, size_t height, size_t depth)
{
    size_t size = 0;

    assert(format == ABCDK_TORCH_TENFMT_NCHW || format == ABCDK_TORCH_TENFMT_NHWC);
    assert(block > 0 && width > 0 && height > 0 && depth > 0);

    if (format == ABCDK_TORCH_TENFMT_NCHW)
    {
        assert(stride >= width);

        size = block * depth * height * stride;
    }
    else if (format == ABCDK_TORCH_TENFMT_NHWC)
    {
        assert(stride >= width * depth);

        size = block * height * stride;
    }

    return size;
}

size_t abcdk_torch_tensor_offset(int format, size_t block, size_t width, size_t stride, size_t height, size_t depth, size_t cell,
                                 size_t n, size_t x, size_t y, size_t z)
{

    size_t off = 0;

    assert(format == ABCDK_TORCH_TENFMT_NCHW || format == ABCDK_TORCH_TENFMT_NHWC);
    assert(block > 0 && width > 0 && stride > 0 && height > 0 && depth > 0 && cell);
    assert(n < block && x < width && y < height && z < depth);

    if (format == ABCDK_TORCH_TENFMT_NCHW)
    {
        assert(stride >= width);

        off = x * cell;
        off += y * stride;
        off += z * height * stride;
        off += n * depth * height * stride;
    }
    else if (format == ABCDK_TORCH_TENFMT_NHWC)
    {
        assert(stride >= width * depth);

        off = x * depth * cell;
        off += z * cell;
        off += y * stride;
        off += n * height * stride;
    }

    return off;
}