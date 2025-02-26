/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VCODEC_H
#define ABCDK_CUDA_VCODEC_H

#include "abcdk/util/queue.h"
#include "abcdk/util/option.h"
#include "abcdk/media/packet.h"
#include "abcdk/media/vcodec.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/device.h"
#include "abcdk/cuda/frame.h"


__BEGIN_DECLS

/**
 * 申请。
 *
 * @param [in] cuda_ctx CUDA环境。仅作指针复制，对象关闭时不会释放。
 */
abcdk_media_vcodec_t *abcdk_cuda_vcodec_alloc(CUcontext cuda_ctx);

/** 
 * 启动。
 * 
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_vcodec_start(abcdk_media_vcodec_t *ctx, abcdk_media_vcodec_param_t *param);

/**
 * 编码。
 *
 * @param [in] src 图像帧。NULL(0) 仅获取编码帧。
 *
 * @return 1 有输出，0 无输出，< 0 出错了。
 */
int abcdk_cuda_vcodec_encode(abcdk_media_vcodec_t *ctx, abcdk_media_packet_t **dst, const abcdk_media_frame_t *src);

/**
 * 解码。
 *
 * @param [in] src 数据帧。NULL(0) 仅获取解码图。src->size == 0 是结束帧。
 *
 * @return 1 有输出，0 无输出，< 0 出错了。
 */
int abcdk_cuda_vcodec_decode(abcdk_media_vcodec_t *ctx, abcdk_media_frame_t **dst, const abcdk_media_packet_t *src);

__END_DECLS

#endif // ABCDK_CUDA_VCODEC_H