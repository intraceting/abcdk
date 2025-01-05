/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/ndarray.h"

size_t abcdk_ndarray_size(abcdk_ndarray_t *ndarray)
{
    size_t size = 0;

    assert(ndarray != NULL);
    assert(ndarray->fmt == ABCDK_NDARRAY_NCHW || ndarray->fmt == ABCDK_NDARRAY_NHWC);
    assert(ndarray->block > 0 && ndarray->width > 0 && ndarray->height > 0 && ndarray->depth > 0);
    assert(ndarray->stride >= ndarray->width && ndarray->cell > 0);

    if (ndarray->fmt == ABCDK_NDARRAY_NCHW)
    {
        size = ndarray->block * ndarray->depth * ndarray->height * ndarray->stride;
    }
    else if (ndarray->fmt == ABCDK_NDARRAY_NHWC)
    {
        size = ndarray->block * ndarray->height * ndarray->stride;
    }

    return size;
}

void abcdk_ndarray_set_stride(abcdk_ndarray_t *ndarray,size_t align)
{
    assert(ndarray != NULL);
    assert(ndarray->fmt == ABCDK_NDARRAY_NCHW || ndarray->fmt == ABCDK_NDARRAY_NHWC);
    assert(ndarray->block > 0 && ndarray->width > 0 && ndarray->height > 0 && ndarray->depth > 0);
    assert(ndarray->cell > 0);

    if (ndarray->fmt == ABCDK_NDARRAY_NCHW)
    {
        ndarray->stride = abcdk_align(ndarray->width * ndarray->cell, align);
    }
    else if (ndarray->fmt == ABCDK_NDARRAY_NHWC)
    {
        ndarray->stride = abcdk_align(ndarray->width * ndarray->depth * ndarray->cell, align);
    }
}

size_t abcdk_ndarray_offset(abcdk_ndarray_t *ndarray, size_t n, size_t x, size_t y, size_t z,int flag)
{
    size_t off = SIZE_MAX;

    assert(ndarray != NULL);
    assert(ndarray->fmt == ABCDK_NDARRAY_NCHW || ndarray->fmt == ABCDK_NDARRAY_NHWC);
    assert(ndarray->block > 0 && ndarray->width > 0 && ndarray->height > 0 && ndarray->depth > 0);
    assert(ndarray->stride >= ndarray->width && ndarray->cell > 0);
    assert(n < ndarray->block && x < ndarray->width && y < ndarray->height && z < ndarray->depth);

    if (flag == ABCDK_NDARRAY_FLIP_H)
    {
        off = abcdk_ndarray_offset(ndarray, n, x, ndarray->height - y - 1, z, 0);
    }
    else if (flag == ABCDK_NDARRAY_FLIP_V)
    {
        off = abcdk_ndarray_offset(ndarray, n, ndarray->width - x - 1, y, z, 0);
    }
    else
    {
        if (ndarray->fmt == ABCDK_NDARRAY_NCHW)
        {
            off = x * ndarray->cell;
            off += y * ndarray->stride;
            off += z * ndarray->height * ndarray->stride;
            off += n * ndarray->depth * ndarray->height * ndarray->stride;
        }
        else if (ndarray->fmt == ABCDK_NDARRAY_NHWC)
        {
            off = x * ndarray->depth * ndarray->cell;
            off += z * ndarray->cell;
            off += y * ndarray->stride;
            off += n * ndarray->height * ndarray->stride;
        }
    }

    return off;
}
