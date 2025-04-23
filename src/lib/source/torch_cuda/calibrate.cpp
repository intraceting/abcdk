/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/calibrate.h"
#include "abcdk/torch/nvidia.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

double abcdk_torch_calibrate_estimate_2d_cuda(abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size, int count, abcdk_torch_image_t *img[], float camera_matrix[3][3], float dist_coeffs[5])
{
    return 1.0;
}

#else // __cuda_cuda_h__

double abcdk_torch_calibrate_estimate_2d_cuda(abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size, int count, abcdk_torch_image_t *img[], float camera_matrix[3][3], float dist_coeffs[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return 1.0;
}

#endif //__cuda_cuda_h__

__END_DECLS
