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

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

__BEGIN_DECLS


/**创建帧图。 */
AVFrame *abcdk_cuda_avframe_alloc(int width,int height,enum AVPixelFormat pixfmt,int align);

/**
 * 图像复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avimage_copy(uint8_t *dst_datas[4], int dst_strides[4],
                            const uint8_t *src_datas[4], const int src_strides[4],
                            int width, int height, enum AVPixelFormat pixfmt,
                            int dst_in_host, int src_in_host);

/**
 * 帧图复制。
 * 
 * @note 仅图像数据。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_copy(AVFrame *dst, const AVFrame *src, int dst_in_host, int src_in_host);

/**
 * 帧图克隆。
 * 
 * @note 仅图像数据。
 * 
 * @return 0 成功，< 0 失败。
 */
AVFrame *abcdk_cuda_avframe_clone(const AVFrame *src, int dst_in_host, int src_in_host);

/**
 * 帧图转换。
 * 
 * @note 仅图像数据。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_cuda_avframe_convert(AVFrame *dst, const AVFrame *src, int dst_in_host, int src_in_host);

__END_DECLS

#endif //AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__

#endif //ABCDK_CUDA_AVUTIL_H