/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_TENSORPROC_H
#define ABCDK_CUDA_TENSORPROC_H

#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"

#ifdef __cuda_cuda_h__

__BEGIN_DECLS

/**
 * 张量值转换。
 *
 * @note dst[z] = ((src[z] / scale[z]) - mean[z]) / std[z];
 *
 * @param [in] dst_packed 目标图的像素排列方式。0 平面，!0 交叉。
 * @param [in] src_packed 源图的像素排列方式。0 平面，!0 交叉。
 * @param [in] scale 系数。
 * @param [in] mean 均值。
 * @param [in] std 方差。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_blob_8u_to_32f_1R(int dst_packed, float *dst, size_t dst_ws,
                                            int src_packed, uint8_t *src, size_t src_ws,
                                            size_t w, size_t h, float scale[1], float mean[1], float std[1]);

/**
 * 张量值转换。
 *
 * @note dst[z] = ((src[z] / scale[z]) - mean[z]) / std[z];
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_blob_8u_to_32f_3R(int dst_packed, float *dst, size_t dst_ws,
                                            int src_packed, uint8_t *src, size_t src_ws,
                                            size_t w, size_t h, float scale[3], float mean[3], float std[3]);

/**
 * 张量值转换。
 *
 * @note dst[z] = ((src[z] * std[z]) + mean[z]) * scale[z];
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_blob_32f_to_8u_1R(int dst_packed, uint8_t *dst, size_t dst_ws,
                                            int src_packed, float *src, size_t src_ws,
                                            size_t w, size_t h, float scale[1], float mean[1], float std[1]);

/**
 * 张量值转换。
 *
 * @note dst[z] = ((src[z] * std[z]) + mean[z]) * scale[z];
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_tensorproc_blob_32f_to_8u_3R(int dst_packed, uint8_t *dst, size_t dst_ws,
                                            int src_packed, float *src, size_t src_ws,
                                            size_t w, size_t h, float scale[3], float mean[3], float std[3]);

__END_DECLS

#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_TENSORPROC_H