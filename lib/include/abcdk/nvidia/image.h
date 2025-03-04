/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_IMAGE_H
#define ABCDK_NVIDIA_IMAGE_H

#include "abcdk/torch/image.h"
#include "abcdk/nvidia/nvidia.h"
#include "abcdk/nvidia/memory.h"
#include "abcdk/nvidia/imgproc.h"
#include "abcdk/nvidia/imgutil.h"

__BEGIN_DECLS


/**申请。 */
abcdk_torch_image_t *abcdk_cuda_image_alloc();

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_cuda_image_reset(abcdk_torch_image_t **ctx, int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_torch_image_t *abcdk_cuda_image_create(int width, int height, int pixfmt, int align);

/**
 * 复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_image_copy(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src);

/**复制。 */
void abcdk_cuda_image_copy_plane(abcdk_torch_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride);

/**克隆。*/
abcdk_torch_image_t *abcdk_cuda_image_clone(int dst_in_host, const abcdk_torch_image_t *src);

/**
 * 帧图格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_image_convert(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src);

/**
 * 帧图保存到文件。
 * 
 * @note 在没有第三方支持的情况下仅支持BMP格式或RAW格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_image_save(const char *dst, const abcdk_torch_image_t *src);

/**
 * 从文件加载。
 * 
 * @param [in] gray 是否加载为灰度图。0 否，!0 是。
 * 
 * @return 0 成功，< 0 失败。
*/
abcdk_torch_image_t *abcdk_cuda_image_load(const char *src, int gray);


__END_DECLS

#endif // ABCDK_NVIDIA_IMAGE_H