/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_JCODEC_H
#define ABCDK_TORCH_JCODEC_H

#include "abcdk/util/object.h"
#include "abcdk/torch/image.h"

__BEGIN_DECLS

/**JPEG编/解码器参数。*/
typedef struct _abcdk_torch_jcodec_param
{
    /**
     * 质量。
     *
     * 1~99 值越大越清晰，占用的空间越多。
     */
    int quality;

} abcdk_torch_jcodec_param_t;

/**JPEG编/解码器。*/
typedef struct _abcdk_torch_jcodec
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

} abcdk_torch_jcodec_t;

/**释放。*/
void abcdk_torch_jcodec_free_host(abcdk_torch_jcodec_t **ctx);

/**释放。*/
void abcdk_torch_jcodec_free_cuda(abcdk_torch_jcodec_t **ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_jcodec_free abcdk_torch_jcodec_free_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_jcodec_free abcdk_torch_jcodec_free_host
#endif //

/** 申请。*/
abcdk_torch_jcodec_t *abcdk_torch_jcodec_alloc_host(int encoder);

/** 申请。*/
abcdk_torch_jcodec_t *abcdk_torch_jcodec_alloc_cuda(int encoder);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_jcodec_alloc abcdk_torch_jcodec_alloc_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_jcodec_alloc abcdk_torch_jcodec_alloc_host
#endif //

/** 
 * 启动。
 * 
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_jcodec_start_host(abcdk_torch_jcodec_t *ctx, abcdk_torch_jcodec_param_t *param);

/** 
 * 启动。
 * 
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_jcodec_start_cuda(abcdk_torch_jcodec_t *ctx, abcdk_torch_jcodec_param_t *param);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_jcodec_start abcdk_torch_jcodec_start_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_jcodec_start abcdk_torch_jcodec_start_host
#endif //

/**编码。 */
abcdk_object_t *abcdk_torch_jcodec_encode_host(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src);

/**编码。 */
abcdk_object_t *abcdk_torch_jcodec_encode_cuda(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_jcodec_encode abcdk_torch_jcodec_encode_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_jcodec_encode abcdk_torch_jcodec_encode_host
#endif //

/**
 * 编码。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_jcodec_encode_to_file_host(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src);

/**
 * 编码。
 *
 * @return 0 成功，< 0  失败。
 */
int abcdk_torch_jcodec_encode_to_file_cuda(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_jcodec_encode_to_file abcdk_torch_jcodec_encode_to_file_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_jcodec_encode_to_file abcdk_torch_jcodec_encode_to_file_host
#endif //


/**解码。 */
abcdk_torch_image_t *abcdk_torch_jcodec_decode_host(abcdk_torch_jcodec_t *ctx, const void *src, int src_size);

/**解码。 */
abcdk_torch_image_t *abcdk_torch_jcodec_decode_cuda(abcdk_torch_jcodec_t *ctx, const void *src, int src_size);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_jcodec_decode abcdk_torch_jcodec_decode_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_jcodec_decode abcdk_torch_jcodec_decode_host
#endif //

/**解码。 */
abcdk_torch_image_t *abcdk_torch_jcodec_decode_from_file_host(abcdk_torch_jcodec_t *ctx, const char *src);

/**解码。 */
abcdk_torch_image_t *abcdk_torch_jcodec_decode_from_file_cuda(abcdk_torch_jcodec_t *ctx, const char *src);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_jcodec_decode_from_file abcdk_torch_jcodec_decode_from_file_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_jcodec_decode_from_file abcdk_torch_jcodec_decode_from_file_host
#endif //


__END_DECLS

#endif // ABCDK_TORCH_JCODEC_H