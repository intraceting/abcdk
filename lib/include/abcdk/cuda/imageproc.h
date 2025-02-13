/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_CUDA_IMAGEPROC_H
#define ABCDK_CUDA_IMAGEPROC_H

#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"

#ifdef HAVE_CUDA

__BEGIN_DECLS


/** 
 * 图像填充。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_image_stuff_8u_c1r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar);

/** 
 * 图像填充。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_image_stuff_8u_c3r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[3]);

/** 
 * 图像填充。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_image_stuff_8u_c4r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[4]);


__END_DECLS

#endif //HAVE_CUDA

#endif //ABCDK_CUDA_IMAGEPROC_H