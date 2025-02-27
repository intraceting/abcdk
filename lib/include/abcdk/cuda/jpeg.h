/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_JPEG_H
#define ABCDK_CUDA_JPEG_H

#include "abcdk/util/option.h"
#include "abcdk/util/object.h"
#include "abcdk/media/packet.h"
#include "abcdk/media/jcodec.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/device.h"
#include "abcdk/cuda/frame.h"


__BEGIN_DECLS

/**
 * 申请。
 *
 * @param [in] cuda_ctx CUDA环境。仅作指针复制，对象关闭时不会释放。
 */
abcdk_media_jcodec_t *abcdk_cuda_jpeg_create(CUcontext cuda_ctx);

/** 
 * 启动。
 * 
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_vcodec_start(abcdk_media_jcodec_t *ctx, abcdk_media_jpeg_param_t *param);

/**编码。 */
abcdk_media_packet_t *abcdk_cuda_jpeg_encode(abcdk_media_jcodec_t *ctx, const abcdk_media_frame_t *src);

/**
 * 编码。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_jpeg_encode_to_file(abcdk_media_jcodec_t *ctx, const char *dst, const abcdk_media_frame_t *src);

/**解码。 */
abcdk_media_frame_t *abcdk_cuda_jpeg_decode(abcdk_media_jcodec_t *ctx, const void *src, int src_size);

/**解码。 */
abcdk_media_frame_t *abcdk_cuda_jpeg_decode_from_file(abcdk_media_jcodec_t *ctx, const void *src);

/**保存。*/
int abcdk_cuda_jpeg_save(const char *dst, const abcdk_media_frame_t *src, CUcontext cuda_ctx);

/**加载。*/
abcdk_media_frame_t *abcdk_cuda_jpeg_load(const char *src, CUcontext cuda_ctx);

__END_DECLS


#endif // ABCDK_CUDA_JPEG_H