/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_CALIBRATE_H
#define ABCDK_TORCH_CALIBRATE_H

#include "abcdk/util/object.h"
#include "abcdk/torch/torch.h"
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/imgcode.h"
#include "abcdk/torch/tensor.h"

__BEGIN_DECLS

/**
 * 标定评估。
 *
 * @param [in] board_size 板子尺寸(行*列)。
 * @param [in] grid_size 格子尺寸(毫米)。
 * @param [out] camera_matrix 内参矩阵。R,T.
 * @param [out] dist_coeffs 畸变系数。k1,k2,p1,p2,k3.
 *
 * @return RMS。
 */
double abcdk_torch_calibrate_estimate_2d_host(abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size, int count, abcdk_torch_image_t *img[], float camera_matrix[3][3], float dist_coeffs[5]);

/**
 * 标定评估。
 *
 * @param [in] grid_size 板子尺寸(行*列)。
 * @param [in] square_size 方格尺寸(毫米)。
 * @param [out] camera_matrix 内参矩阵。R,T.
 * @param [out] dist_coeffs 畸变系数。k1,k2,p1,p2,k3.
 *
 * @return RMS。
 */
double abcdk_torch_calibrate_estimate_2d_cuda(abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size, int count, abcdk_torch_image_t *img[], float camera_matrix[3][3], float dist_coeff[5]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_calibrate_estimate_2d abcdk_torch_calibrate_estimate_2d_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_calibrate_estimate_2d abcdk_torch_calibrate_estimate_2d_host
#endif //

__END_DECLS

#endif // ABCDK_TORCH_CALIBRATE_H