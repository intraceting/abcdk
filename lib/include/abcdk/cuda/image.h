/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_IMAGE_H
#define ABCDK_CUDA_IMAGE_H

#include "abcdk/media/image.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"
#include "abcdk/cuda/imgproc.h"
#include "abcdk/cuda/imgutil.h"

__BEGIN_DECLS


/**申请。 */
abcdk_media_image_t *abcdk_cuda_image_alloc();

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_cuda_image_reset(abcdk_media_image_t **ctx, int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_media_image_t *abcdk_cuda_image_create(int width, int height, int pixfmt, int align);

/**
 * 复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_image_copy(abcdk_media_image_t *dst, const abcdk_media_image_t *src);

/**复制。 */
void abcdk_cuda_image_copy_plane(abcdk_media_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride);

/**克隆。*/
abcdk_media_image_t *abcdk_cuda_image_clone(int dst_in_host, const abcdk_media_image_t *src);

/**
 * 帧图格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_image_convert(abcdk_media_image_t *dst, const abcdk_media_image_t *src);

/**
 * 帧图保存到文件。
 * 
 * @note 仅支持BMP格式，所有非BGR32格式自动换转BGR32格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_image_save(const char *dst, const abcdk_media_image_t *src);


__END_DECLS

#endif // ABCDK_CUDA_IMAGE_H