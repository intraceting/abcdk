/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMGUTIL_H
#define ABCDK_TORCH_IMGUTIL_H

#include "abcdk/util/general.h"
#include "abcdk/ffmpeg/avutil.h"
#include "abcdk/torch/torch.h"
#include "abcdk/torch/pixfmt.h"
#include "abcdk/torch/memory.h"

__BEGIN_DECLS

/**
 * 计算图像每个图层的高度。
 *
 * @param pixfmt 像素格式
 * @param height 高(像素)
 *
 * @return > 0 成功(图层数量)， <= 0 失败。
 */
int abcdk_torch_imgutil_fill_height(int heights[4], int height, int pixfmt);

/**
 * 计算图像每个图层的宽步长(字节)。
 *
 * @param width 宽(像素)
 * @param align 对齐(字节)
 *
 * @return > 0 成功(图层数量)， <= 0 失败。
 */
int abcdk_torch_imgutil_fill_stride(int stride[4], int width, int pixfmt, int align);

/**
 * 分派存储空间。
 *
 * @param buffer 内存指针，传入NULL(0)。
 *
 * @return >0 成功(分派的内存大小)， <= 0 失败。
 */
int abcdk_torch_imgutil_fill_pointer(uint8_t *data[4], const int stride[4], int height, int pixfmt, void *buffer);

/**
 * 计算需要的内存大小。
 *
 * @return >0 成功(需要的内存大小)， <= 0 失败。
 */
int abcdk_torch_imgutil_size(const int stride[4], int height, int pixfmt);

/**
 * 计算需要的内存大小。
 *
 * @return >0 成功(需要的内存大小)， <= 0 失败。
 */
int abcdk_torch_imgutil_size2(int width, int height, int pixfmt, int align);

/** 选取颜色。*/
uint8_t abcdk_torch_imgutil_select_color(int idx, int channel);

/**
 * 图像复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgutil_copy_host(uint8_t *dst_data[4], int dst_stride[4], int dst_in_host,
                                  const uint8_t *src_data[4], const int src_stride[4], int src_in_host,
                                  int width, int height, int pixfmt);

/**
 * 图像复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgutil_copy_cuda(uint8_t *dst_data[4], int dst_stride[4], int dst_in_host,
                                  const uint8_t *src_data[4], const int src_stride[4], int src_in_host,
                                  int width, int height, int pixfmt);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgutil_copy abcdk_torch_imgutil_copy_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgutil_copy abcdk_torch_imgutil_copy_host
#endif //


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
int abcdk_torch_imgutil_blob_8u_to_32f_host(int dst_packed, float *dst, size_t dst_ws,
                                            int src_packed, uint8_t *src, size_t src_ws,
                                            size_t b, size_t w, size_t h, size_t c,
                                            float scale[], float mean[], float std[]);

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
int abcdk_torch_imgutil_blob_8u_to_32f_cuda(int dst_packed, float *dst, size_t dst_ws,
                                            int src_packed, uint8_t *src, size_t src_ws,
                                            size_t b, size_t w, size_t h, size_t c, 
                                            float scale[], float mean[], float std[]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgutil_blob_8u_to_32f abcdk_torch_imgutil_blob_8u_to_32f_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgutil_blob_8u_to_32f abcdk_torch_imgutil_blob_8u_to_32f_host
#endif //

/**
 * 值转换。
 *
 * @note dst[z] = ((src[z] * std[z]) + mean[z]) * scale[z];
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgutil_blob_32f_to_8u_host(int dst_packed, uint8_t *dst, size_t dst_ws,
                                            int src_packed, float *src, size_t src_ws,
                                            size_t b, size_t w, size_t h, size_t c,
                                            float scale[], float mean[], float std[]);

/**
 * 值转换。
 *
 * @note dst[z] = ((src[z] * std[z]) + mean[z]) * scale[z];
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgutil_blob_32f_to_8u_cuda(int dst_packed, uint8_t *dst, size_t dst_ws,
                                            int src_packed, float *src, size_t src_ws,
                                            size_t b, size_t w, size_t h, size_t c,
                                            float scale[], float mean[], float std[]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgutil_blob_32f_to_8u abcdk_torch_imgutil_blob_32f_to_8u_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgutil_blob_32f_to_8u abcdk_torch_imgutil_blob_32f_to_8u_host
#endif //


__END_DECLS

#endif // ABCDK_TORCH_IMGUTIL_H