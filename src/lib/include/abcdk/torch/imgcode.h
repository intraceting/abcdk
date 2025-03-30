/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMGCODE_H
#define ABCDK_TORCH_IMGCODE_H

#include "abcdk/torch/image.h"
#include "abcdk/torch/jcodec.h"

__BEGIN_DECLS


/**
 * 保存到文件。
 * 
 * @note 仅支持JPEG格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgcode_save_host(const char *dst, const abcdk_torch_image_t *src);

/**
 * 保存到文件。
 * 
 * @note 仅支持JPEG格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgcode_save_cuda(const char *dst, const abcdk_torch_image_t *src);


#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgcode_save abcdk_torch_imgcode_save_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgcode_save abcdk_torch_imgcode_save_host
#endif //

/**
 * 从文件加载。
 * 
 * @note 仅支持JPEG格式。
 * 
 * @return 0 成功，< 0 失败。
*/
abcdk_torch_image_t *abcdk_torch_imgcode_load_host(const char *src);

/**
 * 从文件加载。
 * 
 * @note 仅支持JPEG格式。
 * 
 * @return 0 成功，< 0 失败。
*/
abcdk_torch_image_t *abcdk_torch_imgcode_load_cuda(const char *src);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgcode_load abcdk_torch_imgcode_load_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgcode_load abcdk_torch_imgcode_load_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_IMGCODE_H