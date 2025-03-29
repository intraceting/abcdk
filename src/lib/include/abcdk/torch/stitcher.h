/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_STITCHER_H
#define ABCDK_TORCH_STITCHER_H

#include "abcdk/util/object.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/opencv.h"


__BEGIN_DECLS

/**简单的全景拼接引擎。*/
typedef struct _abcdk_torch_stitcher
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

} abcdk_torch_stitcher_t;

/**销毁。 */
void abcdk_torch_stitcher_destroy_host(abcdk_torch_stitcher_t **ctx);

/**销毁。 */
void abcdk_torch_stitcher_destroy_cuda(abcdk_torch_stitcher_t **ctx);


#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_destroy abcdk_torch_stitcher_destroy_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_destroy abcdk_torch_stitcher_destroy_host
#endif //

/**创建。 */
abcdk_torch_stitcher_t *abcdk_torch_stitcher_create_host();

/**创建。 */
abcdk_torch_stitcher_t *abcdk_torch_stitcher_create_cuda();


#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_create abcdk_torch_stitcher_create_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_create abcdk_torch_stitcher_create_host
#endif //


/**
 * 保存元数据。
 *
 * @param [in] magic 魔法字符串。用于区别元数据的所有者。
 *
 * @return 0 成功，< 0 失败。
 */
abcdk_object_t *abcdk_torch_stitcher_metadata_dump_host(abcdk_torch_stitcher_t *ctx, const char *magic);

/**
 * 保存元数据。
 *
 * @param [in] magic 魔法字符串。用于区别元数据的所有者。
 *
 * @return 0 成功，< 0 失败。
 */
abcdk_object_t *abcdk_torch_stitcher_metadata_dump_cuda(abcdk_torch_stitcher_t *ctx, const char *magic);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_metadata_dump abcdk_torch_stitcher_metadata_dump_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_metadata_dump abcdk_torch_stitcher_metadata_dump_host
#endif //

/**
 * 加载元数据。
 *
 * @return 0 成功， < 0 失败，-127 魔法字符串验证失败。
 */
int abcdk_torch_stitcher_metadata_load_host(abcdk_torch_stitcher_t *ctx, const char *magic, const char *data);

/**
 * 加载元数据。
 *
 * @return 0 成功， < 0 失败，-127 魔法字符串验证失败。
 */
int abcdk_torch_stitcher_metadata_load_cuda(abcdk_torch_stitcher_t *ctx, const char *magic, const char *data);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_metadata_load abcdk_torch_stitcher_metadata_load_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_metadata_load abcdk_torch_stitcher_metadata_load_host
#endif //

/**
 * 设特征发现算法。
 *
 * @param [in] name 名称。目前仅支持ORB、SIFT、SURF。默认：ORB
 *
 * @return 0 成功，-1 失败(恢复默认)。
 */
int abcdk_torch_stitcher_set_feature_host(abcdk_torch_stitcher_t *ctx, const char *name);

/**
 * 设特征发现算法。
 *
 * @param [in] name 名称。目前仅支持ORB、SIFT、SURF。默认：ORB
 *
 * @return 0 成功，-1 失败(恢复默认)。
 */
int abcdk_torch_stitcher_set_feature_cuda(abcdk_torch_stitcher_t *ctx, const char *name);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_set_feature abcdk_torch_stitcher_set_feature_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_set_feature abcdk_torch_stitcher_set_feature_host
#endif //

/**
 * 评估。
 *
 * @note 仅支持主机内存对象。
 *
 * @param [in] img 分屏图像的数组。顺序无关，评估时自动排序。
 * @param [in] mask 掩码图像的数组，顺序与分屏图像对应。元素的值为NULL(0)时，忽略。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_stitcher_estimate_host(abcdk_torch_stitcher_t *ctx, int count, abcdk_torch_image_t *img[], abcdk_torch_image_t *mask[], float good_threshold);

/**
 * 评估。
 *
 * @note 仅支持主机内存对象。
 *
 * @param [in] img 分屏图像的数组。顺序无关，评估时自动排序。
 * @param [in] mask 掩码图像的数组，顺序与分屏图像对应。元素的值为NULL(0)时，忽略。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_stitcher_estimate_cuda(abcdk_torch_stitcher_t *ctx, int count, abcdk_torch_image_t *img[], abcdk_torch_image_t *mask[], float good_threshold);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_estimate abcdk_torch_stitcher_estimate_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_estimate abcdk_torch_stitcher_estimate_host
#endif //

/**
 * 设图像变换算法。
 *
 * @param [in] name 名称。目前仅支持plane、spherical。默认：spherical
 *
 * @return 0 成功，-1 失败(恢复默认)。
 */
int abcdk_torch_stitcher_set_warper_host(abcdk_torch_stitcher_t *ctx, const char *name);

/**
 * 设图像变换算法。
 *
 * @param [in] name 名称。目前仅支持plane、spherical。默认：spherical
 *
 * @return 0 成功，-1 失败(恢复默认)。
 */
int abcdk_torch_stitcher_set_warper_cuda(abcdk_torch_stitcher_t *ctx, const char *name);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_set_warper abcdk_torch_stitcher_set_warper_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_set_warper abcdk_torch_stitcher_set_warper_host
#endif //

/**
 * 构建全景参数。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_stitcher_build_host(abcdk_torch_stitcher_t *ctx);

/**
 * 构建全景参数。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_stitcher_build_cuda(abcdk_torch_stitcher_t *ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_build abcdk_torch_stitcher_build_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_build abcdk_torch_stitcher_build_host
#endif //

/**
 * 全景融合。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_stitcher_compose_host(abcdk_torch_stitcher_t *ctx, abcdk_torch_image_t *out, int count, abcdk_torch_image_t *img[]);

/**
 * 全景融合。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_stitcher_compose_cuda(abcdk_torch_stitcher_t *ctx, abcdk_torch_image_t *out, int count, abcdk_torch_image_t *img[]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_stitcher_compose abcdk_torch_stitcher_compose_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_stitcher_compose abcdk_torch_stitcher_compose_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_STITCHER_H
