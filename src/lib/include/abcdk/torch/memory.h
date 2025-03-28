/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_MEMORY_H
#define ABCDK_TORCH_MEMORY_H

#include "abcdk/torch/torch.h"

__BEGIN_DECLS

/** 释放。*/
void abcdk_torch_free_host(void **data);

/** 释放。*/
void abcdk_torch_free_cuda(void **data);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_free abcdk_torch_free_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_free abcdk_torch_free_host
#endif //

/** 申请。 */
void *abcdk_torch_alloc_host(size_t size);

/** 申请。 */
void *abcdk_torch_alloc_cuda(size_t size);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_alloc abcdk_torch_alloc_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_alloc abcdk_torch_alloc_host
#endif //

/**
 * 申请。
 *
 * @note 如果申请成功，则全部赋值为零。
 */
void *abcdk_torch_alloc_z_host(size_t size);

/**
 * 申请。
 *
 * @note 如果申请成功，则全部赋值为零。
 */
void *abcdk_torch_alloc_z_cuda(size_t size);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_alloc_z abcdk_torch_alloc_z_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_alloc_z abcdk_torch_alloc_z_host
#endif //

/** 赋值。*/
void *abcdk_torch_memset_host(void *dst, int val, size_t size);

/** 赋值。*/
void *abcdk_torch_memset_cuda(void *dst, int val, size_t size);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_memset abcdk_torch_memset_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_memset abcdk_torch_memset_host
#endif //

/**
 * 复制(1D)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_memcpy_host(void *dst, int dst_in_host, const void *src, int src_in_host, size_t size);

/**
 * 复制(1D)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_memcpy_cuda(void *dst, int dst_in_host, const void *src, int src_in_host, size_t size);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_memcpy abcdk_torch_memcpy_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_memcpy abcdk_torch_memcpy_host
#endif //

/**
 * 复制(2D)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_memcpy_2d_host(void *dst, size_t dst_pitch, size_t dst_x_bytes, size_t dst_y, int dst_in_host,
                               const void *src, size_t src_pitch, size_t src_x_bytes, size_t src_y, int src_in_host,
                               size_t roi_width_bytes, size_t roi_height);

/**
 * 复制(2D)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_memcpy_2d_cuda(void *dst, size_t dst_pitch, size_t dst_x_bytes, size_t dst_y, int dst_in_host,
                               const void *src, size_t src_pitch, size_t src_x_bytes, size_t src_y, int src_in_host,
                               size_t roi_width_bytes, size_t roi_height);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_memcpy_2d abcdk_torch_memcpy_2d_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_memcpy_2d abcdk_torch_memcpy_2d_host
#endif //

/** 克隆。*/
void *abcdk_torch_copyfrom_host(const void *src, size_t size, int src_in_host);

/** 克隆。*/
void *abcdk_torch_copyfrom_cuda(const void *src, size_t size, int src_in_host);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_copyfrom abcdk_torch_copyfrom_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_copyfrom abcdk_torch_copyfrom_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_MEMORY_H