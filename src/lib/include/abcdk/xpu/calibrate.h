/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_CALIBRATE_H
#define ABCDK_XPU_CALIBRATE_H

#include "abcdk/xpu/image.h"
#include "abcdk/util/object.h"
#include "abcdk/util/mmap.h"

__BEGIN_DECLS

/**标定环境.*/
typedef struct _abcdk_xpu_calibrate abcdk_xpu_calibrate_t;

/**释放.*/
void abcdk_xpu_calibrate_free(abcdk_xpu_calibrate_t **ctx);

/**创建.*/
abcdk_xpu_calibrate_t *abcdk_xpu_calibrate_alloc();

/**
 * @param [in] board_cols 格子行数.
 * @param [in] board_rows 格子列数.
 * @param [in] grid_width 格子的宽.
 * @param [in] grid_height 格子的高.
*/
void abcdk_xpu_calibrate_setup(abcdk_xpu_calibrate_t *ctx, int board_cols, int board_rows, int grid_width, int grid_height);

/**
 * 检测角点.
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_xpu_calibrate_detect_corners(abcdk_xpu_calibrate_t *ctx, const abcdk_xpu_image_t *img, int win_width, int win_height);

/**
 * 评估参数.
 * 
 * @return 重投影误差. 数值越小, 表示相机参数越准确.
*/
double abcdk_xpu_calibrate_estimate_parameters(abcdk_xpu_calibrate_t *ctx);

/**
 * 构建参数.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_calibrate_build_parameters(abcdk_xpu_calibrate_t *ctx, double alpha);

/**
 * 保存参数.
 *
 * @param [in] magic 魔法字符串. 用于区别元数据的所有者.
 *
 */
abcdk_object_t *abcdk_xpu_calibrate_dump_parameters(abcdk_xpu_calibrate_t *ctx, const char *magic);

/**
 * 保存参数.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_calibrate_dump_parameters_to_file(abcdk_xpu_calibrate_t *ctx,const char *dst, const char *magic);

/**
 * 加载参数.
 *
 * @return 0 成功, < 0 失败, -127 魔法字符串验证失败.
 */
int abcdk_xpu_calibrate_load_parameters(abcdk_xpu_calibrate_t *ctx, const char *src, const char *magic);

/**
 * 加载参数.
 *
 * @return 0 成功, < 0 失败, -127 魔法字符串验证失败.
 */
int abcdk_xpu_calibrate_load_parameters_from_file(abcdk_xpu_calibrate_t *ctx, const char *src, const char *magic);

/**
 * 畸变矫正.
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_calibrate_undistort(abcdk_xpu_calibrate_t *ctx, const abcdk_xpu_image_t*src, abcdk_xpu_image_t **dst, abcdk_xpu_inter_t inter_mode);

__END_DECLS

#endif // ABCDK_XPU_CALIBRATE_H