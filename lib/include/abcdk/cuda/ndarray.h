/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_NDARRAY_H
#define ABCDK_CUDA_NDARRAY_H

#include "abcdk/util/ndarray.h"
#include "abcdk/cuda/memory.h"

#ifdef __cuda_cuda_h__

__BEGIN_DECLS


/**获取多维数组的内存类型。*/
CUmemorytype abcdk_cuda_ndarray_memory_type(const abcdk_ndarray_t *src);

/**创建多维数组。 */
abcdk_ndarray_t *abcdk_cuda_ndarray_alloc(int fmt, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

/**
 * 多维数组复制。
 *
 * @note 仅数组数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_ndarray_copy(abcdk_ndarray_t *dst, const abcdk_ndarray_t *src);

/**
 * 多维数组克隆。
 *
 * @note 仅数组数据。
 */
abcdk_ndarray_t *abcdk_cuda_ndarray_clone(int dst_in_host, const abcdk_ndarray_t *src);

/**
 * 多维数组克隆。
 *
 * @note 仅数组数据。
 */
abcdk_ndarray_t *abcdk_cuda_ndarray_clone2(int dst_in_host,
                                           const uint8_t *src_data, const int src_stride, int src_in_host,
                                           int fmt, size_t block, size_t width, size_t height, size_t depth, size_t cell);

__END_DECLS

#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_NDARRAY_H