/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "util/ndarray.h"

size_t abcdk_ndarray_offset(abcdk_ndarray_t *ndarray, size_t n, size_t x, size_t y, size_t z,int flag)
{
    size_t off = SIZE_MAX;

    assert(ndarray != NULL);
    assert(ndarray->fmt == ABCDK_NDARRAY_NCHW || ndarray->fmt == ABCDK_NDARRAY_NHWC);
    assert(ndarray->blocks > 0 && ndarray->width > 0 && ndarray->height > 0 && ndarray->depth > 0);
    assert(ndarray->width_bytes >= ndarray->width && ndarray->cell_bytes > 0);
    assert(n < ndarray->blocks && x < ndarray->width && y < ndarray->height && z < ndarray->depth);

    if (flag == ABCDK_NDARRAY_FLIP_H)
    {
        off = abcdk_ndarray_offset(ndarray, n, x, ndarray->height - y - 1, z, 0);
    }
    else if (flag == ABCDK_NDARRAY_FLIP_V)
    {
        off = abcdk_ndarray_offset(ndarray, n, ndarray->width - x - 1, y, z, 0);
    }
    else if(flag == ABCDK_NDARRAY_ROTATE_C)
    {
        off = abcdk_ndarray_offset(ndarray, n, ndarray->height - y - 1, x, z, 0);
    }
    else if(flag == ABCDK_NDARRAY_ROTATE_AC)
    {

    }
    else
    {
        if (ndarray->fmt == ABCDK_NDARRAY_NCHW)
        {
            off = x * ndarray->cell_bytes;
            off += y * ndarray->width_bytes;
            off += z * ndarray->height * ndarray->width_bytes;
            off += n * ndarray->depth * ndarray->height * ndarray->width_bytes;
        }
        else if (ndarray->fmt == ABCDK_NDARRAY_NHWC)
        {
            off = x * ndarray->depth * ndarray->cell_bytes;
            off += z * ndarray->cell_bytes;
            off += y * ndarray->width_bytes;
            off += n * ndarray->height * ndarray->width_bytes;
        }
    }

    return off;
}
