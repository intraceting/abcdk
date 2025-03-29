/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_TENSOR_H
#define ABCDK_TORCH_TENSOR_H

#include "abcdk/torch/torch.h"
#include "abcdk/torch/tenutil.h"
#include "abcdk/torch/memory.h"

__BEGIN_DECLS

/**简单的张量。*/
typedef struct _abcdk_torch_tensor
{
    /**数据指针。*/
    uint8_t *data;

    /**格式。*/
    int format;

    /**块数量。*/
    size_t block;

    /**宽(每块)。*/
    size_t width;
    
    /**宽步长(字节)。*/
    size_t stride;

    /**高(每块)。*/
    size_t height;

    /**深(每块)。*/
    size_t depth;

    /**单元尺寸(字节)。*/
    size_t cell;

    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

} abcdk_torch_tensor_t;


/**释放。*/
void abcdk_torch_tensor_free_host(abcdk_torch_tensor_t **ctx);

/**释放。*/
void abcdk_torch_tensor_free_cuda(abcdk_torch_tensor_t **ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_tensor_free abcdk_torch_tensor_free_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_tensor_free abcdk_torch_tensor_free_host
#endif //


/**申请。*/
abcdk_torch_tensor_t *abcdk_torch_tensor_alloc_host();

/**申请。*/
abcdk_torch_tensor_t *abcdk_torch_tensor_alloc_cuda();

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_tensor_alloc abcdk_torch_tensor_alloc_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_tensor_alloc abcdk_torch_tensor_alloc_host
#endif //

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_tensor_reset_host(abcdk_torch_tensor_t **ctx, int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_tensor_reset_cuda(abcdk_torch_tensor_t **ctx, int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_tensor_reset abcdk_torch_tensor_reset_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_tensor_reset abcdk_torch_tensor_reset_host
#endif //

/**创建。*/
abcdk_torch_tensor_t *abcdk_torch_tensor_create_host(int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

/**创建。*/
abcdk_torch_tensor_t *abcdk_torch_tensor_create_cuda(int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_tensor_create abcdk_torch_tensor_create_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_tensor_create abcdk_torch_tensor_create_host
#endif //

/** 复制。 */
int abcdk_torch_tensor_copy_host(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src);

/** 复制。 */
int abcdk_torch_tensor_copy_cuda(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_tensor_copy abcdk_torch_tensor_copy_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_tensor_copy abcdk_torch_tensor_copy_host
#endif //

/** 复制。 */
int abcdk_torch_tensor_copy_block_host(abcdk_torch_tensor_t *dst, int dst_block, const uint8_t *src_data, int src_stride);

/** 复制。 */
int abcdk_torch_tensor_copy_block_cuda(abcdk_torch_tensor_t *dst, int dst_block, const uint8_t *src_data, int src_stride);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_tensor_copy_block abcdk_torch_tensor_copy_block_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_tensor_copy_block abcdk_torch_tensor_copy_block_host
#endif //

/** 克隆。 */
abcdk_torch_tensor_t *abcdk_torch_tensor_clone_host(int dst_in_host, const abcdk_torch_tensor_t *src);

/** 克隆。 */
abcdk_torch_tensor_t *abcdk_torch_tensor_clone_cuda(int dst_in_host, const abcdk_torch_tensor_t *src);


#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_tensor_clone abcdk_torch_tensor_clone_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_tensor_clone abcdk_torch_tensor_clone_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_TENSOR_H