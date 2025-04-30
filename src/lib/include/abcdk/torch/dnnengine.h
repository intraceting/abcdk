/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_DNN_ENGINE_H
#define ABCDK_TORCH_DNN_ENGINE_H

#include "abcdk/util/option.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/imgutil.h"
#include "abcdk/torch/dnn.h"

__BEGIN_DECLS


/**DNN引擎环境。*/
typedef struct _abcdk_torch_dnn_engine
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

} abcdk_torch_dnn_engine_t;

/**释放。*/
void abcdk_torch_dnn_engine_free_host(abcdk_torch_dnn_engine_t **ctx);

/**释放。*/
void abcdk_torch_dnn_engine_free_cuda(abcdk_torch_dnn_engine_t **ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_dnn_engine_free abcdk_torch_dnn_engine_free_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_dnn_engine_free abcdk_torch_dnn_engine_free_host
#endif //

/**申请。*/
abcdk_torch_dnn_engine_t *abcdk_torch_dnn_engine_alloc_host();

/**申请。*/
abcdk_torch_dnn_engine_t *abcdk_torch_dnn_engine_alloc_cuda();

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_dnn_engine_alloc abcdk_torch_dnn_engine_alloc_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_dnn_engine_alloc abcdk_torch_dnn_engine_alloc_host
#endif //

/**
 * 加载模型。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_engine_load_model_host(abcdk_torch_dnn_engine_t *ctx, const char *file, abcdk_option_t *opt);

/**
 * 加载模型。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_engine_load_model_cuda(abcdk_torch_dnn_engine_t *ctx, const char *file, abcdk_option_t *opt);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_dnn_engine_load_model abcdk_torch_dnn_engine_load_model_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_dnn_engine_load_model abcdk_torch_dnn_engine_load_model_host
#endif //

/**
 * 获取张量信息。
 *
 * @return 数量。
 */
int abcdk_torch_dnn_engine_fetch_tensor_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_dnn_tensor tensor[]);

/**
 * 获取张量信息。
 *
 * @return 数量。
 */
int abcdk_torch_dnn_engine_fetch_tensor_cuda(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_dnn_tensor tensor[]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_dnn_engine_fetch_tensor abcdk_torch_dnn_engine_fetch_tensor_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_dnn_engine_fetch_tensor abcdk_torch_dnn_engine_fetch_tensor_host
#endif //

/**
 * 推理。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_engine_infer_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_image_t *img[]);

/**
 * 推理。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_engine_infer_cuda(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_image_t *img[]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_dnn_engine_infer abcdk_torch_dnn_engine_infer_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_dnn_engine_infer abcdk_torch_dnn_engine_infer_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_DNN_ENGINE_H
