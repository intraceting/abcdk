/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_FRAME_H
#define ABCDK_CUDA_FRAME_H

#include "abcdk/media/frame.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"
#include "abcdk/cuda/imgproc.h"
#include "abcdk/cuda/imgutil.h"

__BEGIN_DECLS


/**申请。 */
abcdk_media_frame_t *abcdk_cuda_frame_alloc();

/**
 * 重置。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_frame_reset(abcdk_media_frame_t *ctx,int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_media_frame_t *abcdk_cuda_frame_create(int width, int height, int pixfmt, int align);

/**
 * 帧图保存到文件。
 * 
 * @note 仅支持BMP格式，所有非BGR32格式自动换转BGR32格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_frame_save(const char *dst, const abcdk_media_frame_t *src);

/**
 * 复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_frame_copy(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src);

/** 克隆。*/
abcdk_media_frame_t *abcdk_cuda_frame_clone(int dst_in_host, const abcdk_media_frame_t *src);

/**
 * 克隆。
 *
 * @note 仅图像数据。
 */
abcdk_media_frame_t *abcdk_cuda_frame_clone2(int dst_in_host, const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt);

/**
 * 帧图格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_frame_convert(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src);

__END_DECLS

#endif // ABCDK_CUDA_FRAME_H