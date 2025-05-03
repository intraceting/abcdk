/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMGPROC_H
#define ABCDK_TORCH_IMGPROC_H

#include "abcdk/util/trace.h"
#include "abcdk/util/geometry.h"
#include "abcdk/torch/torch.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/pixfmt.h"
#include "abcdk/torch/imgutil.h"

__BEGIN_DECLS

/**
 * 图像填充。
 *
 * @param [in] roi 感兴趣区域。NULL(0) 全部。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_stuff_host(abcdk_torch_image_t *dst, uint32_t scalar[], const abcdk_torch_rect_t *roi);

/**
 * 图像填充。
 *
 * @param [in] roi 感兴趣区域。NULL(0) 全部。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_stuff_cuda(abcdk_torch_image_t *dst, uint32_t scalar[], const abcdk_torch_rect_t *roi);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_stuff abcdk_torch_imgproc_stuff_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_stuff abcdk_torch_imgproc_stuff_host
#endif //

/**
 * 全景图像融合(从左到右)。
 *
 * @param [in out] panorama 全景图像。
 * @param [in] compose 融合图像。
 * @param [in] scalar 填充色。
 * @param [in] overlap_x  融合图像在全景图像的左上角X坐标。
 * @param [in] overlap_y  融合图像在全景图像的左上角Y坐标。
 * @param [in] overlap_w  融合图像在全景图像中重叠宽度。
 * @param [in] optimize_seam 接缝美化。0 禁用，!0 启用。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_compose_host(abcdk_torch_image_t *panorama, abcdk_torch_image_t *compose,
                                     uint32_t scalar[], size_t overlap_x, size_t overlap_y, size_t overlap_w,
                                     int optimize_seam);

/**
 * 全景图像融合(从左到右)。
 *
 * @param [in out] panorama 全景图像。
 * @param [in] compose 融合图像。
 * @param [in] scalar 填充色。
 * @param [in] overlap_x  融合图像在全景图像的左上角X坐标。
 * @param [in] overlap_y  融合图像在全景图像的左上角Y坐标。
 * @param [in] overlap_w  融合图像在全景图像中重叠宽度。
 * @param [in] optimize_seam 接缝美化。0 禁用，!0 启用。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_compose_cuda(abcdk_torch_image_t *panorama, abcdk_torch_image_t *compose,
                                     uint32_t scalar[], size_t overlap_x, size_t overlap_y, size_t overlap_w,
                                     int optimize_seam);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_compose abcdk_torch_imgproc_compose_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_compose abcdk_torch_imgproc_compose_host
#endif //

/**
 * 调整亮度。
 *
 * @note dst[z] = src[z] * alpha[z] + bate[z]
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_brightness_host(abcdk_torch_image_t *dst, float alpha[], float bate[]);

/**
 * 调整亮度。
 *
 * @note dst[z] = src[z] * alpha[z] + bate[z]
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_brightness_cuda(abcdk_torch_image_t *dst, float alpha[], float bate[]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_brightness abcdk_torch_imgproc_brightness_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_brightness abcdk_torch_imgproc_brightness_host
#endif //

/**
 * 暗通道除雾。
 *
 * @note 建议：a=220,m=0.35,w=0.9
 *
 * @param [in] dack_a 暗通道像素值。
 * @param [in] dack_m 模数。
 * @param [in] dack_w 权重。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_defog_host(abcdk_torch_image_t *dst, uint32_t dack_a, float dack_m, float dack_w);

/**
 * 暗通道除雾。
 *
 * @note 建议：a=220,m=0.35,w=0.9
 *
 * @param [in] dack_a 暗通道像素值。
 * @param [in] dack_m 模数。
 * @param [in] dack_w 权重。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_defog_cuda(abcdk_torch_image_t *dst, uint32_t dack_a, float dack_m, float dack_w);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_defog abcdk_torch_imgproc_defog_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_defog abcdk_torch_imgproc_defog_host
#endif //

/**
 * 画矩形框。
 *
 * @param [in] corner 左上(x1,y1)，右下(x2,y2)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_drawrect_host(abcdk_torch_image_t *dst, uint32_t color[], int weight, int corner[4]);

/**
 * 画矩形框。
 *
 * @param [in] corner 左上(x1,y1)，右下(x2,y2)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_imgproc_drawrect_cuda(abcdk_torch_image_t *dst, uint32_t color[], int weight, int corner[4]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_drawrect abcdk_torch_imgproc_drawrect_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_drawrect abcdk_torch_imgproc_drawrect_host
#endif //

/**
 * 缩放。
 *
 * @param [in] keep_aspect_ratio 保持纵横比例。
 * @param [in] inter_mode 插值方案。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgproc_resize_host(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                    const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                    int keep_aspect_ratio, int inter_mode);

/**
 * 缩放。
 *
 * @param [in] keep_aspect_ratio 保持纵横比例。
 * @param [in] inter_mode 插值方案。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgproc_resize_cuda(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                    const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                    int keep_aspect_ratio, int inter_mode);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_resize abcdk_torch_imgproc_resize_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_resize abcdk_torch_imgproc_resize_host
#endif //

/**
 * 变换。
 *
 * @param [in] dst_quad 目标角点。 [0][] 左上，[1][] 右上，[2][] 右下，[3][]左下。
 * @param [in] src_quad 源图角点。 [0][] 左上，[1][] 右上，[2][] 右下，[3][]左下。
 * @param [in] warp_mode 变换模式。1 透视，2 仿射。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgproc_warp_host(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi, const abcdk_torch_point_t dst_quad[4],
                                  const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi, const abcdk_torch_point_t src_quad[4],
                                  int warp_mode, int inter_mode);
/**
 * 变换。
 *
 * @param [in] dst_quad 目标角点。 [0][] 左上，[1][] 右上，[2][] 右下，[3][]左下。
 * @param [in] src_quad 源图角点。 [0][] 左上，[1][] 右上，[2][] 右下，[3][]左下。
 * @param [in] warp_mode 变换模式。1 透视，2 仿射。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgproc_warp_cuda(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi, const abcdk_torch_point_t dst_quad[4],
                                  const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi, const abcdk_torch_point_t src_quad[4],
                                  int warp_mode, int inter_mode);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_warp abcdk_torch_imgproc_warp_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_warp abcdk_torch_imgproc_warp_host
#endif //

/**
 * 重映射。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgproc_remap_host(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                   const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                   const abcdk_torch_image_t *xmap, const abcdk_torch_image_t *ymap,
                                   int inter_mode);

/**
 * 重映射。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgproc_remap_cuda(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                   const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                   const abcdk_torch_image_t *xmap, const abcdk_torch_image_t *ymap,
                                   int inter_mode);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_remap abcdk_torch_imgproc_remap_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_remap abcdk_torch_imgproc_remap_host
#endif //

/**
 * 构建畸变矫正表。
 *
 * @param [out] xmap 水平矫正表。
 * @param [out] ymap 垂直矫正表。
 * @param [in] size 图像尺寸。
 * @param [in] alpha 图像的裁剪系数，介于0(无黑边)和1(无黑边)之间。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_imgproc_undistort_buildmap_host(abcdk_torch_image_t **xmap, abcdk_torch_image_t **ymap,
                                                abcdk_torch_size_t *size, double alpha,
                                                const double camera_matrix[3][3], const double dist_coeffs[5]);

/**
 * 构建畸变矫正表。
 *
 * @param [out] xmap 水平矫正表。
 * @param [out] ymap 垂直矫正表。
 * @param [in] size 图像尺寸。
 * @param [in] alpha 图像的裁剪系数，介于0(无黑边)和1(无黑边)之间。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_imgproc_undistort_buildmap_cuda(abcdk_torch_image_t **xmap, abcdk_torch_image_t **ymap,
                                                abcdk_torch_size_t *size, double alpha,
                                                const double camera_matrix[3][3], const double dist_coeffs[5]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_undistort_buildmap abcdk_torch_imgproc_undistort_buildmap_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_undistort_buildmap abcdk_torch_imgproc_undistort_buildmap_host
#endif //

/**
 * 画线段。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgproc_line_host(abcdk_torch_image_t *dst, const abcdk_torch_point_t *p1, const abcdk_torch_point_t *p2, uint32_t color[], int weight);

/**
 * 画线段。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_torch_imgproc_line_cuda(abcdk_torch_image_t *dst, const abcdk_torch_point_t *p1, const abcdk_torch_point_t *p2, uint32_t color[], int weight);


#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_imgproc_line abcdk_torch_imgproc_line_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_imgproc_line abcdk_torch_imgproc_line_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_IMGPROC_H