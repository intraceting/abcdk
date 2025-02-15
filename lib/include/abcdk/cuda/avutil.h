/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_AVUTIL_H
#define ABCDK_CUDA_AVUTIL_H

#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"
#include "abcdk/ffmpeg/avutil.h"
#include "abcdk/ffmpeg/swscale.h"
#include "abcdk/util/geometry.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

__BEGIN_DECLS

/**创建帧图。 */
AVFrame *abcdk_cuda_avframe_alloc(int width, int height, enum AVPixelFormat pixfmt, int align);

/**
 * 图像复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avimage_copy(uint8_t *dst_datas[4], int dst_strides[4], int dst_in_host,
                            const uint8_t *src_datas[4], const int src_strides[4], int src_in_host,
                            int width, int height, enum AVPixelFormat pixfmt);

/**
 * 帧图复制。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_copy(AVFrame *dst, int dst_in_host, const AVFrame *src, int src_in_host);

/**
 * 帧图克隆。
 *
 * @note 仅图像数据。
 *
 */
AVFrame *abcdk_cuda_avframe_clone(int dst_in_host, const AVFrame *src, int src_in_host);

/**
 * 帧图格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_convert(AVFrame *dst, int dst_in_host, const AVFrame *src, int src_in_host);

/**
 * 帧图缩放。
 *
 * @note 仅图像数据。
 *
 * @param [in] keep_aspect_ratio 保持纵横比例。
 * @param [in] inter_mode 插值方案。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_resize(AVFrame *dst, const NppiRect *dst_roi, int dst_in_host,
                              const AVFrame *src, const NppiRect *src_roi, int src_in_host,
                              int keep_aspect_ratio, NppiInterpolationMode inter_mode);

/**
 * 帧图透视变换。
 *
 * @param [in] quad 角点。 [0][] 左上，[1][] 右上，[2][] 右下，[3][]左下。
 *
 * @param [in] back 变换方向。0 矩形向多边型变换。!0 多边型向矩形变换。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_warp_perspective(AVFrame *dst, const NppiRect *dst_roi, int dst_in_host,
                                        const AVFrame *src, const NppiRect *src_roi, int src_in_host,
                                        const NppiPoint quad[4], const NppiRect *quad_roi,
                                        int back, NppiInterpolationMode inter_mode);

__END_DECLS

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_AVUTIL_H