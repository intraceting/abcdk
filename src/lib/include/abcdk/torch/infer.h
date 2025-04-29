/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_INFER_H
#define ABCDK_TORCH_INFER_H

#include "abcdk/util/option.h"
#include "abcdk/torch/tensorproc.h"
#include "abcdk/torch/image.h"

__BEGIN_DECLS

/**推理张量。*/
typedef struct _abcdk_torch_infer_tensor
{
    /**索引。*/
    int index;

    /**名字。*/
    const char *name;

    /**模式。*/
    int mode;

    /**维度。*/
    abcdk_torch_dims_t dims;

    /**数据。*/
    const void *data;

} abcdk_torch_infer_tensor;

/**推理引擎。*/
typedef struct _abcdk_torch_infer
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

} abcdk_torch_infer_t;

/**释放。*/
void abcdk_torch_infer_free_host(abcdk_torch_infer_t **ctx);

/**释放。*/
void abcdk_torch_infer_free_cuda(abcdk_torch_infer_t **ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_infer_free abcdk_torch_infer_free_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_infer_free abcdk_torch_infer_free_host
#endif //

/**申请。*/
abcdk_torch_infer_t *abcdk_torch_infer_alloc_host();

/**申请。*/
abcdk_torch_infer_t *abcdk_torch_infer_alloc_cuda();

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_infer_alloc abcdk_torch_infer_alloc_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_infer_alloc abcdk_torch_infer_alloc_host
#endif //

/**
 * 加载模型。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_infer_load_model_host(abcdk_torch_infer_t *ctx, const char *file, abcdk_option_t *opt);

/**
 * 加载模型。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_infer_load_model_cuda(abcdk_torch_infer_t *ctx, const char *file, abcdk_option_t *opt);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_infer_load_model abcdk_torch_infer_load_model_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_infer_load_model abcdk_torch_infer_load_model_host
#endif //

/**
 * 推理。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_infer_execute_host(abcdk_torch_infer_t *ctx, int count, abcdk_torch_image_t *img[]);

/**
 * 推理。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_infer_execute_cuda(abcdk_torch_infer_t *ctx, int count, abcdk_torch_image_t *img[]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_infer_execute abcdk_torch_infer_execute_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_infer_execute abcdk_torch_infer_execute_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_INFER_H
