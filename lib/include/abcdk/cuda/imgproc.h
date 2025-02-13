/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_CUDA_IMGPROC_H
#define ABCDK_CUDA_IMGPROC_H

#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"

#ifdef HAVE_CUDA

__BEGIN_DECLS

/**
 * 图像填充。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_imgproc_stuff_8u_c1r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[1]);

/**
 * 图像填充。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_imgproc_stuff_8u_c3r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[3]);

/**
 * 图像填充。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_imgproc_stuff_8u_c4r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[4]);

/**
 * 全景图像融合(从左到右)。
 * 
 * @param [in] panorama_w 全景图像宽度。
 * @param [in] panorama_ws 全景图像宽度步长。
 * @param [in] panorama_h 全景图像高度。
 * 
 * @param [in] compose_w 融合图像宽度。
 * @param [in] compose_ws 融合图像宽度步长。
 * @param [in] compose_h 融合图像高度。
 * 
 * @param [in] overlap_x  融合图像在全景图像的左上角X坐标。
 * @param [in] overlap_y  融合图像在全景图像的左上角Y坐标。
 * @param [in] overlap_w  融合图像在全景图像中重叠宽度。
 * @param [in] optimize_seam 接缝美化。0 禁用，!0 启用。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_imgproc_compose_8u_c1r(uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                      uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                      uint8_t scalar[1], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam);

/**
 * 全景图像融合(从左到右)。
 *
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_imgproc_compose_8u_c3r(uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                      uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                      uint8_t scalar[3], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam);

/**
 * 全景图像融合(从左到右)。
 *
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_imgproc_compose_8u_c4r(uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                      uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                      uint8_t scalar[4], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam);

/**
 * 调整亮度。
 * 
 * @note dst[] = src[] * alpha[] + bate[]
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_imgproc_brightness_8u_c1r(uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                         size_t w, size_t h, float *alpha, float *bate);

/**
 * 调整亮度。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_imgproc_brightness_8u_c3r(uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                         size_t w, size_t h, float *alpha, float *bate);

/**
 * 调整亮度。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_imgproc_brightness_8u_c4r(uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                         size_t w, size_t h, float *alpha, float *bate);

__END_DECLS

#endif //HAVE_CUDA

#endif //ABCDK_CUDA_IMGPROC_H