/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_OPENCV_STITCHER_H
#define ABCDK_OPENCV_STITCHER_H

#include "abcdk/util/object.h"
#include "abcdk/torch/image.h"
#include "abcdk/nvidia/image.h"
#include "abcdk/nvidia/device.h"
#include "abcdk/opencv/opencv.h"

__BEGIN_DECLS

/**简单的全景拼接引擎。*/
typedef struct _abcdk_opencv_stitcher abcdk_opencv_stitcher_t;

/**销毁。 */
void abcdk_opencv_stitcher_destroy(abcdk_opencv_stitcher_t **ctx);

/**创建。 */
abcdk_opencv_stitcher_t *abcdk_opencv_stitcher_create();

/**创建(NVIDIA)。 */
abcdk_opencv_stitcher_t *abcdk_opencv_stitcher_create_cuda(CUcontext cuda_ctx);

/**
 * 保存元数据。
 * 
 * @param [in] magic 魔法字符串。用于区别元数据的所有者。
 *
 * @return 0 成功，< 0 失败。
 */
abcdk_object_t *abcdk_opencv_stitcher_metadata_dump(abcdk_opencv_stitcher_t *ctx, const char *magic);

/**
 * 加载元数据。
 *
 * @return 0 成功， < 0 失败，-127 魔法字符串验证失败。
 */
int abcdk_opencv_stitcher_metadata_load(abcdk_opencv_stitcher_t *ctx, const char *magic, const char *data);

/**
 * 设特征发现算法。
 * 
 * @param [in] name 名称。目前仅支持ORB、SIFT、SURF。默认：ORB
 */
void abcdk_opencv_stitcher_set_feature_finder(abcdk_opencv_stitcher_t *ctx, const char *name);

/**
 * 评估。
 * 
 * @return 0 成功， < 0 失败。
 */
int abcdk_opencv_stitcher_estimate_transform(abcdk_opencv_stitcher_t *ctx,int count, abcdk_torch_image_t *img[], abcdk_torch_image_t *mask[], float good_threshold);

/**
 * 设图像变换算法。
 * 
 * @param [in] name 名称。目前仅支持plane、spherical。默认：spherical
 */
void abcdk_opencv_stitcher_set_warper(abcdk_opencv_stitcher_t *ctx, const char *name);

/**
 * 构建全景参数。
 * 
 * @return 0 成功， < 0 失败。
*/
int abcdk_opencv_stitcher_build_panorama_param(abcdk_opencv_stitcher_t *ctx);

/**
 * 全景融合。
 * 
 * @return 0 成功， < 0 失败。
*/
int abcdk_opencv_stitcher_compose_panorama(abcdk_opencv_stitcher_t *ctx,abcdk_torch_image_t *out, int count, abcdk_torch_image_t *img[]);

__END_DECLS


#endif //ABCDK_OPENCV_STITCHER_H
