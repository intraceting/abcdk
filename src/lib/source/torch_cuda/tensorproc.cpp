/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/tensorproc.h"
#include "abcdk/torch/nvidia.h"
#include "../torch/tensorproc.hxx"
#include "grid.hxx"

#ifdef __cuda_cuda_h__

__BEGIN_DECLS

__END_DECLS

#else //__cuda_cuda_h__

__BEGIN_DECLS

int abcdk_torch_tensorproc_blob_8u_to_32f_cuda(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_tensorproc_blob_32f_to_8u_cuda(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_tensorproc_reshape_cuda(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

__END_DECLS

#endif // __cuda_cuda_h__
