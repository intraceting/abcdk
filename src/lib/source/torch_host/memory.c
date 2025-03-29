/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/memory.h"

void abcdk_torch_free_host(void **data)
{
    abcdk_heap_freep(data);
}

void *abcdk_torch_alloc_host(size_t size)
{
    return abcdk_heap_alloc(size);
}


void *abcdk_torch_alloc_z_host(size_t size)
{
    return abcdk_heap_alloc(size);
}

void *abcdk_torch_memset_host(void *dst, int val, size_t size)
{
    return memset(dst, val, size);
}

int abcdk_torch_memcpy_host(void *dst, int dst_in_host, const void *src, int src_in_host, size_t size)
{
    assert(dst != NULL && dst_in_host != 0 && src != NULL && src_in_host != 0 && size > 0);

    memcpy(dst, src, size);

    return 0;
}

int abcdk_torch_memcpy_2d_host(void *dst, size_t dst_pitch, size_t dst_x_bytes, size_t dst_y, int dst_in_host,
                               const void *src, size_t src_pitch, size_t src_x_bytes, size_t src_y, int src_in_host,
                               size_t roi_width_bytes, size_t roi_height)
{
    assert(dst_in_host != 0 && src_in_host != 0);

    abcdk_memcpy_2d(dst, dst_pitch, dst_x_bytes, dst_y, src, src_pitch, src_x_bytes, src_y, roi_width_bytes, roi_height);

    return 0;
}

void *abcdk_torch_copyfrom_host(const void *src, size_t size, int src_in_host)
{
    assert(src_in_host != 0);

    return abcdk_heap_clone(src,size);
}

