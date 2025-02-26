/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_IMGUTIL_H
#define ABCDK_CUDA_IMGUTIL_H

#include "abcdk/media/imgutil.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"

__BEGIN_DECLS

/**
 * 图像复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_image_copy(uint8_t *dst_data[4], int dst_stride[4], int dst_in_host,
                          const uint8_t *src_data[4], const int src_stride[4], int src_in_host,
                          int width, int height, int pixfmt);

__END_DECLS

#endif //ABCDK_CUDA_IMGUTIL_H
