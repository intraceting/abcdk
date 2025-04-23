/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"

#ifdef __cuda_cuda_h__

__BEGIN_DECLS

int abcdk_torch_imgproc_undistort_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src, const float camera_matrix[3][3], const float dist_coeffs[5])
{
    return -1;
}

__END_DECLS

#endif //__cuda_cuda_h__
