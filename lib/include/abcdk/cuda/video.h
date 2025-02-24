/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_H
#define ABCDK_CUDA_VIDEO_H

#include "abcdk/util/queue.h"
#include "abcdk/util/option.h"
#include "abcdk/util/object.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/device.h"
#include "abcdk/cuda/avutil.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef AVCODEC_AVCODEC_H

__BEGIN_DECLS

/**VIDEO编/解码器。*/
typedef struct _abcdk_cuda_video abcdk_cuda_video_t;

/**释放。*/
void abcdk_cuda_video_destroy(abcdk_cuda_video_t **ctx);

/**
 * 创建。
 *
 * @param [in] cuda_ctx CUDA环境。仅作指针复制，对象关闭时不会释放。
 */
abcdk_cuda_video_t *abcdk_cuda_video_create(int encode, abcdk_option_t *cfg, CUcontext cuda_ctx);

/**
 * 同步。
 *
 * @note 解码：codec_id、extradata、extradata_size输入。
 * @note 编码：framerate、width、height、codec_id输入，extradata、extradata_size输出。
 */
int abcdk_cuda_video_sync(abcdk_cuda_video_t *ctx, AVCodecContext *opt);

/**
 * 编码。
 *
 * @param [in] src 图像帧。NULL(0) 仅获取编码帧。
 *
 * @return 1 有输出，0 无输出，< 0 出错了。
 */
int abcdk_cuda_video_encode(abcdk_cuda_video_t *ctx, AVPacket **dst, const AVFrame *src);

/**
 * 解码。
 *
 * @param [in] src 数据帧。NULL(0) 仅获取解码图。src->size == 0 是结束帧。
 *
 * @return 1 有输出，0 无输出，< 0 出错了。
 */
int abcdk_cuda_video_decode(abcdk_cuda_video_t *ctx, AVFrame **dst, const AVPacket *src);

__END_DECLS

#endif // AVCODEC_AVCODEC_H
#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_H