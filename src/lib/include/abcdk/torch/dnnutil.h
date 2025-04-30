/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_DNN_UTIL_H
#define ABCDK_TORCH_DNN_UTIL_H

#include "abcdk/util/option.h"
#include "abcdk/torch/image.h"

__BEGIN_DECLS

/**
 * 模型加速。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_model_forward_host(const char *dst, const char *src, abcdk_option_t *opt);

/**
 * 模型加速。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_model_forward_cuda(const char *dst, const char *src, abcdk_option_t *opt);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_dnn_model_forward abcdk_torch_dnn_model_forward_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_dnn_model_forward abcdk_torch_dnn_model_forward_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_DNN_UTIL_H
