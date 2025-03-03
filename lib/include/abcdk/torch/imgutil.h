/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMGUTIL_H
#define ABCDK_TORCH_IMGUTIL_H

#include "abcdk/util/general.h"
#include "abcdk/torch/torch.h"
#include "abcdk/torch/pixfmt.h"

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

/**
 * 图像复制。
 */
void abcdk_torch_imgutil_copy(uint8_t *dst_data[4], int dst_stride[4],
                              const uint8_t *src_data[4], const int src_stride[4],
                              int width, int height, int pixfmt);

__END_DECLS

#endif // ABCDK_TORCH_IMGUTIL_H