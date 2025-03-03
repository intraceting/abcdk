/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_TENUTIL_H
#define ABCDK_TORCH_TENUTIL_H

#include "abcdk/torch/torch.h"
#include "abcdk/torch/tenfmt.h"

__BEGIN_DECLS

/**
 * 计算宽度步长(字节)。
 * 
 * @param align 对齐字节。
*/
size_t abcdk_torch_tenutil_stride(int format, size_t width, size_t depth, size_t cell, size_t align);

/**计算占用空间(字节)。*/
size_t abcdk_torch_tenutil_size(int format, size_t block, size_t width, size_t stride, size_t height, size_t depth);


/** 计算坐标的偏移量(字节)。 */
size_t abcdk_torch_tensor_offset(int format, size_t block, size_t width, size_t stride, size_t height, size_t depth, size_t cell,
                                 size_t n, size_t x, size_t y, size_t z);

__END_DECLS

#endif // ABCDK_TORCH_TENUTIL_H