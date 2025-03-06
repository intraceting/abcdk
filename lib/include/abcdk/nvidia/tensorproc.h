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
 * 张量值转换。
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
 * 张量值转换。
 *
 * @note dst[z] = ((src[z] * std[z]) + mean[z]) * scale[z];
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_blob_32f_to_8u(abcdk_torch_tensor_t *dst,const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[]);

/**
 * 存储格式转换。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_reshape_8u(int dst_packed, uint8_t *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                     int src_packed, uint8_t *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c);

/**
 * 存储格式转换。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_reshape_16u(int dst_packed, uint16_t *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                      int src_packed, uint16_t *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c);

/**
 * 存储格式转换。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_reshape_32u(int dst_packed, uint32_t *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                      int src_packed, uint32_t *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c);

/**
 * 存储格式转换。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_reshape_64u(int dst_packed, uint64_t *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                      int src_packed, uint64_t *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c);

/**
 * 存储格式转换。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_reshape_32f(int dst_packed, float *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                      int src_packed, float *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c);

/**
 * 存储格式转换。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_reshape_64f(int dst_packed, double *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                      int src_packed, double *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c);

__END_DECLS


#endif // ABCDK_NVIDIA_TENSORPROC_H