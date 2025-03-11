/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_TENSORPROC_H
#define ABCDK_NVIDIA_TENSORPROC_H

#include "abcdk/nvidia/tensor.h"


__BEGIN_DECLS

/**
 * 值转换。
 *
 * @note dst[z] = ((src[z] / scale[z]) - mean[z]) / std[z];
 *
 * @param [in] scale 系数。
 * @param [in] mean 均值。
 * @param [in] std 方差。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_blob_8u_to_32f(abcdk_torch_tensor_t *dst,const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[]);

/**
 * 值转换。
 *
 * @note dst[z] = ((src[z] * std[z]) + mean[z]) * scale[z];
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_blob_32f_to_8u(abcdk_torch_tensor_t *dst,const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[]);

/**
 * 重塑。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_reshape(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src);


__END_DECLS


#endif // ABCDK_NVIDIA_TENSORPROC_H