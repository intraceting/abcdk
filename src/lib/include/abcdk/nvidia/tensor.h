/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_TENSOR_H
#define ABCDK_NVIDIA_TENSOR_H

#include "abcdk/torch/torch.h"
#include "abcdk/torch/tenutil.h"
#include "abcdk/torch/tensor.h"
#include "abcdk/nvidia/image.h"
#include "abcdk/nvidia/memory.h"

__BEGIN_DECLS

/**申请。*/
abcdk_torch_tensor_t *abcdk_cuda_tensor_alloc();

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_cuda_tensor_reset(abcdk_torch_tensor_t **ctx, int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

/**创建。*/
abcdk_torch_tensor_t *abcdk_cuda_tensor_create(int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

/** 复制。 */
void abcdk_cuda_tensor_copy(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src);

/** 复制。 */
void abcdk_cuda_tensor_copy_block(abcdk_torch_tensor_t *dst, int dst_block, const uint8_t *src_data, int src_stride);

/** 克隆。 */
abcdk_torch_tensor_t *abcdk_cuda_tensor_clone(int dst_in_host, const abcdk_torch_tensor_t *src);


__END_DECLS

#endif // ABCDK_NVIDIA_TENSOR_H