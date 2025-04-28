/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/opencv.h"

__BEGIN_DECLS

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)


int abcdk_torch_infer_model_forward_cuda(const char *dst,const char *src, abcdk_option_t *opt)
{

}

#else // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

int abcdk_torch_infer_model_forward_cuda(const char *dst,const char *src, abcdk_option_t *opt)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或TensorRT工具。"));
    return -1;
}

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

__END_DECLS