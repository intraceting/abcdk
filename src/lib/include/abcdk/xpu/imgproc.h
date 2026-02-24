/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_IMGPROC_H
#define ABCDK_XPU_IMGPROC_H

#include "abcdk/xpu/image.h"
#include "abcdk/ffmpeg/sws.h"

__BEGIN_DECLS

/**
 * 格式转换.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_convert(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst);

/**
 * 缩放.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_resize(const abcdk_xpu_image_t *src, const abcdk_xpu_rect_t *src_roi, abcdk_xpu_image_t *dst, abcdk_xpu_inter_t inter_mode);

/**
 * 图像填充.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_stuff(abcdk_xpu_image_t *dst, const abcdk_xpu_rect_t *roi, const abcdk_xpu_scalar_t *scalar);

/**
 * 调整亮度.
 *
 * @note dst[z] = src[z] * alpha.f32[z] + bate.f32[z]
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_brightness(abcdk_xpu_image_t *dst, const abcdk_xpu_scalar_t *alpha, const abcdk_xpu_scalar_t *bate);

/**
 * 画矩形框.
 *
 * @param [in] corner 左上(x1,y1), 右下(x2,y2).
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_rectangle(abcdk_xpu_image_t *dst, const abcdk_xpu_rect_t *rect, int weight, const abcdk_xpu_scalar_t *color);

/**
 * 自由变换.
 *
 * @param [in] warp_mode 变换模式.1 透视, 2 仿射.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_warp(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst, const abcdk_xpu_matrix_3x3_t *coeffs, int warp_mode, abcdk_xpu_inter_t inter_mode);

/**
 * 四边形到四边形变换.
 * 
 * @note 仿射变换仅支持平行四边形角点, 否则可能产生预料之外的结果.
 *
 * @param [in] src_quad 源图角点. [0][] 左上, [1][] 右上, [2][] 右下, [3][]左下.
 * @param [in] dst_quad 目标角点. [0][] 左上, [1][] 右上, [2][] 右下, [3][]左下.
 * @param [in] warp_mode 变换模式.1 透视, 2 仿射.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_warp_quad2quad(const abcdk_xpu_image_t *src, const abcdk_xpu_point_t src_quad[4], abcdk_xpu_image_t *dst, const abcdk_xpu_point_t dst_quad[4], int warp_mode, abcdk_xpu_inter_t inter_mode);

/**
 * 重映射.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_remap(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst, const abcdk_xpu_image_t *xmap, const abcdk_xpu_image_t *ymap, abcdk_xpu_inter_t inter_mode);

/**
 * 构建畸变矫正表.
 *
 * @param [in] size 图像尺寸.
 * @param [in] alpha 图像的裁剪系数, 介于0(无黑边)和1(无黑边)之间.
 * @param [out] xmap 水平矫正表.
 * @param [out] ymap 垂直矫正表.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_undistort(const abcdk_xpu_size_t *size, double alpha,
                            const abcdk_xpu_matrix_3x3_t *camera_matrix,
                            const abcdk_xpu_scalar_t *dist_coeffs,
                            abcdk_xpu_image_t **xmap, abcdk_xpu_image_t **ymap);

/**
 * 画线段.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_line(abcdk_xpu_image_t *dst, const abcdk_xpu_point_t *p1, const abcdk_xpu_point_t *p2,
                       const abcdk_xpu_scalar_t *color, int weight);

/**
 * 画掩码.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_mask(abcdk_xpu_image_t *dst, const abcdk_xpu_image_t *feature, float threshold, const abcdk_xpu_scalar_t *color, int less_or_not);

/**
 * 四边形2矩形.
 *
 * @param [in] src_quad 源图角点. [0][] 左上, [1][] 右上, [2][] 右下, [3][] 左下.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_quad2rect(const abcdk_xpu_image_t *src, const abcdk_xpu_point_t src_quad[4], abcdk_xpu_image_t *dst, abcdk_xpu_inter_t inter_mode);

/**
 * 人脸扣图并矫正.
 * 
 * @param [in] face_kpt 五官角点. [0][] 眼(右), [1][] 眼(左), [2][] 鼻, [3][] 嘴(右), [4][] 嘴(左).
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_imgproc_face_warp(const abcdk_xpu_image_t *src, const abcdk_xpu_point_t face_kpt[5], abcdk_xpu_image_t **dst, abcdk_xpu_inter_t inter_mode);

__END_DECLS

#endif // ABCDK_XPU_IMGPROC_H