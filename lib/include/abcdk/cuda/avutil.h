/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_AVUTIL_H
#define ABCDK_CUDA_AVUTIL_H

#include "abcdk/ffmpeg/avformat.h"
#include "abcdk/ffmpeg/avcodec.h"
#include "abcdk/ffmpeg/avutil.h"
#include "abcdk/ffmpeg/swscale.h"
#include "abcdk/util/geometry.h"
#include "abcdk/image/bmp.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"
#include "abcdk/cuda/imgproc.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

__BEGIN_DECLS

/**
 * 图像复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avimage_copy(uint8_t *dst_datas[4], int dst_strides[4], int dst_in_host,
                            const uint8_t *src_datas[4], const int src_strides[4], int src_in_host,
                            int width, int height, enum AVPixelFormat pixfmt);

/**获取帧图的内存类型。*/
CUmemorytype abcdk_cuda_avframe_memory_type(const AVFrame *src);

/**创建帧图。 */
AVFrame *abcdk_cuda_avframe_alloc(int width, int height, enum AVPixelFormat pixfmt, int align);

/**
 * 帧图复制。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_copy(AVFrame *dst, const AVFrame *src);

/**
 * 帧图克隆。
 *
 * @note 仅图像数据。
 *
 */
AVFrame *abcdk_cuda_avframe_clone(int dst_in_host, const AVFrame *src);

/**
 * 帧图保存到文件。
 * 
 * @note 仅支持BMP格式，所有非BGR32格式自动换转BGR32格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_save(const char *dst, const AVFrame *src);

/**
 * 帧图格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_convert(AVFrame *dst, const AVFrame *src);

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
int abcdk_cuda_avframe_resize(AVFrame *dst, const NppiRect *dst_roi,
                              const AVFrame *src, const NppiRect *src_roi,
                              int keep_aspect_ratio, NppiInterpolationMode inter_mode);

/**
 * 帧图变换。
 *
 * @param [in] dst_quad 目标角点。 [0][] 左上，[1][] 右上，[2][] 右下，[3][]左下。
 * @param [in] src_quad 源图角点。 [0][] 左上，[1][] 右上，[2][] 右下，[3][]左下。
 * @param [in] warp_mode 变换模式。1 透视，2 仿射。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_warp(AVFrame *dst, const NppiRect *dst_roi, const NppiPoint dst_quad[4],
                            const AVFrame *src, const NppiRect *src_roi, const NppiPoint src_quad[4],
                            int warp_mode, NppiInterpolationMode inter_mode);

/**
 * 帧图重映射。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_remap(AVFrame *dst, const NppiRect *dst_roi,
                             const AVFrame *src, const NppiRect *src_roi,
                             const AVFrame *xmap, const AVFrame *ymap,
                             NppiInterpolationMode inter_mode);

/**
 * 全景图像融合(从左到右)。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_avframe_compose(AVFrame *panorama, const AVFrame *compose,
                               uint8_t scalar[4], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam);

__END_DECLS

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_AVUTIL_H