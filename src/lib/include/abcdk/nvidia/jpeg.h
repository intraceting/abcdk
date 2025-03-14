/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_JPEG_H
#define ABCDK_NVIDIA_JPEG_H

#include "abcdk/util/object.h"
#include "abcdk/torch/jcodec.h"
#include "abcdk/torch/packet.h"
#include "abcdk/nvidia/nvidia.h"
#include "abcdk/nvidia/device.h"
#include "abcdk/nvidia/image.h"


__BEGIN_DECLS

/**
 * 申请。
 *
 * @param [in] cuda_ctx CUDA环境。仅作指针复制，对象关闭时不会释放。
 */
abcdk_torch_jcodec_t *abcdk_cuda_jpeg_create(int encoder, CUcontext cuda_ctx);

/** 
 * 启动。
 * 
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_jpeg_start(abcdk_torch_jcodec_t *ctx, abcdk_torch_jcodec_param_t *param);

/**编码。 */
abcdk_object_t *abcdk_cuda_jpeg_encode(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src);

/**
 * 编码。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_cuda_jpeg_encode_to_file(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src);

/**解码。 */
abcdk_torch_image_t *abcdk_cuda_jpeg_decode(abcdk_torch_jcodec_t *ctx, const void *src, int src_size);

/**解码。 */
abcdk_torch_image_t *abcdk_cuda_jpeg_decode_from_file(abcdk_torch_jcodec_t *ctx, const void *src);

/**保存。*/
int abcdk_cuda_jpeg_save(const char *dst, const abcdk_torch_image_t *src, CUcontext cuda_ctx);

/**加载。*/
abcdk_torch_image_t *abcdk_cuda_jpeg_load(const char *src, CUcontext cuda_ctx);

__END_DECLS


#endif // ABCDK_NVIDIA_JPEG_H