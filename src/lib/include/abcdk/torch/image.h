/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMAGE_H
#define ABCDK_TORCH_IMAGE_H

#include "abcdk/torch/torch.h"
#include "abcdk/torch/memory.h"
#include "abcdk/torch/imgutil.h"

__BEGIN_DECLS

/**媒体图像结构。*/
typedef struct _abcdk_torch_image
{
    /**图层指针。 */
    uint8_t *data[4];

    /**图层步长。 */
    int stride[4];

    /**图像格式。 */
    int pixfmt;

    /**宽(像素)。*/
    int width;

    /**高(像素)。*/
    int height;

    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

}abcdk_torch_image_t;

/**释放。*/
void abcdk_torch_image_free_host(abcdk_torch_image_t **ctx);

/**释放。*/
void abcdk_torch_image_free_cuda(abcdk_torch_image_t **ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_free abcdk_torch_image_free_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_free abcdk_torch_image_free_host
#endif //

/**申请。*/
abcdk_torch_image_t *abcdk_torch_image_alloc_host();

/**申请。*/
abcdk_torch_image_t *abcdk_torch_image_alloc_cuda();

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_alloc abcdk_torch_image_alloc_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_alloc abcdk_torch_image_alloc_host
#endif //


/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_image_reset_host(abcdk_torch_image_t **ctx, int width, int height, int pixfmt, int align);

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_image_reset_cuda(abcdk_torch_image_t **ctx, int width, int height, int pixfmt, int align);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_reset abcdk_torch_image_reset_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_reset abcdk_torch_image_reset_host
#endif //

/**创建。*/
abcdk_torch_image_t *abcdk_torch_image_create_host(int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_torch_image_t *abcdk_torch_image_create_cuda(int width, int height, int pixfmt, int align);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_create abcdk_torch_image_create_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_create abcdk_torch_image_create_host
#endif //

/** 复制。 */
int abcdk_torch_image_copy_host(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src);

/** 复制。 */
int abcdk_torch_image_copy_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_copy abcdk_torch_image_copy_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_copy abcdk_torch_image_copy_host
#endif //

/** 复制。 */
int abcdk_torch_image_copy_plane_host(abcdk_torch_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride);

/** 复制。 */
int abcdk_torch_image_copy_plane_cuda(abcdk_torch_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_copy_plane abcdk_torch_image_copy_plane_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_copy_plane abcdk_torch_image_copy_plane_host
#endif //

/**克隆。*/
abcdk_torch_image_t *abcdk_torch_image_clone_host(int dst_in_host, const abcdk_torch_image_t *src);

/**克隆。*/
abcdk_torch_image_t *abcdk_torch_image_clone_cuda(int dst_in_host, const abcdk_torch_image_t *src);


#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_clone abcdk_torch_image_clone_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_clone abcdk_torch_image_clone_host
#endif //

/**
 * 格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_image_convert_host(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src);

/**
 * 格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_image_convert_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_convert abcdk_torch_image_convert_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_convert abcdk_torch_image_convert_host
#endif //

/**
 * 保存(RAW)。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_image_dump_host(const char *dst, const abcdk_torch_image_t *src);

/**
 * 保存(RAW)。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_image_dump_cuda(const char *dst, const abcdk_torch_image_t *src);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_image_dump abcdk_torch_image_dump_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_image_dump abcdk_torch_image_dump_host
#endif //


__END_DECLS

#endif // ABCDK_TORCH_IMAGE_H