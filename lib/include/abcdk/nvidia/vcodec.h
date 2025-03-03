/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_VCODEC_H
#define ABCDK_NVIDIA_VCODEC_H

#include "abcdk/util/queue.h"
#include "abcdk/torch/vcodec.h"
#include "abcdk/nvidia/nvidia.h"
#include "abcdk/nvidia/device.h"
#include "abcdk/nvidia/image.h"


__BEGIN_DECLS

/**
 * 申请。
 *
 * @param [in] cuda_ctx CUDA环境。仅作指针复制，对象关闭时不会释放。
 */
abcdk_torch_vcodec_t *abcdk_cuda_vcodec_alloc(int encoder,CUcontext cuda_ctx);

/** 
 * 启动。
 * 
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_vcodec_start(abcdk_torch_vcodec_t *ctx, abcdk_torch_vcodec_param_t *param);

/**
 * 编码。
 *
 * @param [in] src 图像。NULL(0) 仅获取编码帧。
 *
 * @return 1 有输出，0 无输出，< 0 出错了。
 */
int abcdk_cuda_vcodec_encode(abcdk_torch_vcodec_t *ctx, abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src);

/**
 * 解码。
 *
 * @param [in] src_data 数据包指针。NULL(0) 仅获取解码图。
 * @param [in] src_size 数据包长度。0 是结束帧。
 * @param [in] src_pts 播放时间。一个递增的时间值，影响解码图的输出顺序。
 *
 * @return 1 有输出，0 无输出，< 0 出错了。
 */
int abcdk_cuda_vcodec_decode(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const abcdk_torch_packet_t *src);

__END_DECLS

#endif // ABCDK_NVIDIA_VCODEC_H