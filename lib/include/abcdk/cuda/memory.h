/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_MEMORY_H
#define ABCDK_CUDA_MEMORY_H

#include "abcdk/cuda/cuda.h"


__BEGIN_DECLS

/** 内存释放。*/
void abcdk_cuda_free(void **data);

/**
 * 内存申请。
 *
 */
void *abcdk_cuda_alloc(size_t size);

/**
 * 内存申请。
 *
 * @note 如果申请成功，则全部赋值为零。
 */
void *abcdk_cuda_alloc_z(size_t size);

/** 内存赋值。*/
void *abcdk_cuda_memset(void *dst, int val, size_t size);

/**
 * 内存复制(1D)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_memcpy(void *dst, int dst_in_host, const void *src, int src_in_host, size_t size);

/**
 * 内存复制(2D)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_memcpy_2d(void *dst, size_t dst_pitch, size_t dst_x_bytes, size_t dst_y, int dst_in_host,
                         const void *src, size_t src_pitch, size_t src_x_bytes, size_t src_y, int src_in_host,
                         size_t roi_width_bytes, size_t roi_height);

/** 内存克隆。*/
void *abcdk_cuda_copyfrom(const void *src, size_t size, int src_in_host);

__END_DECLS



#endif // ABCDK_CUDA_MEMORY_H