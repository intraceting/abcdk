/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_STITCHER_H
#define ABCDK_XPU_STITCHER_H

#include "abcdk/xpu/image.h"
#include "abcdk/util/object.h"
#include "abcdk/util/mmap.h"

__BEGIN_DECLS

/**拼接环境.*/
typedef struct _abcdk_xpu_stitcher abcdk_xpu_stitcher_t;

/**释放.*/
void abcdk_xpu_stitcher_free(abcdk_xpu_stitcher_t **ctx);

/**创建. */
abcdk_xpu_stitcher_t *abcdk_xpu_stitcher_alloc();

/**
 * 设置特征发现算法.
 *
 * @param [in] name 算法名称. 目前仅支持ORB, SIFT, SURF. 默认: ORB
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_stitcher_set_feature_finder(abcdk_xpu_stitcher_t *ctx, const char *name);

/**
 * 设置图像变换算法.
 *
 * @param [in] name 算法名称. 目前仅支持plane, spherical. 默认: spherical
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_stitcher_set_warper(abcdk_xpu_stitcher_t *ctx, const char *name);

/**
 * 评估参数.
 *
 * @param [in] img 分屏图像的数组, 评估时自动排序.
 * @param [in] mask 掩码图像的数组, 顺序与分屏图像对应, 允许为NULL(0).
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_stitcher_estimate_parameters(abcdk_xpu_stitcher_t *ctx, int count,const abcdk_xpu_image_t *img[], const abcdk_xpu_image_t *mask[], float threshold);

/**
 * 构建参数.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_stitcher_build_parameters(abcdk_xpu_stitcher_t *ctx);

/**
 * 保存参数.
 *
 * @param [in] magic 魔法字符串. 用于区别元数据的所有者.
 *
 */
abcdk_object_t *abcdk_xpu_stitcher_dump_parameters(abcdk_xpu_stitcher_t *ctx, const char *magic);

/**
 * 保存参数.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_stitcher_dump_parameters_to_file(abcdk_xpu_stitcher_t *ctx,const char *dst, const char *magic);

/**
 * 加载参数.
 *
 * @return 0 成功, < 0 失败, -127 魔法字符串验证失败.
 */
int abcdk_xpu_stitcher_load_parameters(abcdk_xpu_stitcher_t *ctx, const char *src, const char *magic);

/**
 * 加载参数.
 *
 * @return 0 成功, < 0 失败, -127 魔法字符串验证失败.
 */
int abcdk_xpu_stitcher_load_parameters_from_file(abcdk_xpu_stitcher_t *ctx, const char *src, const char *magic);

/**
 * 图像融合.
 *
 * @note 待拼接图像顺序必须与评估时保持一致.
 *
 * @return @return 0 成功, < 0 失败.
 */
int abcdk_xpu_stitcher_compose(abcdk_xpu_stitcher_t *ctx, int count, const abcdk_xpu_image_t *img[], abcdk_xpu_image_t **out, int optimize_seam);

__END_DECLS

#endif // ABCDK_XPU_STITCHER_H