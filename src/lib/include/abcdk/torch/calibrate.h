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

__BEGIN_DECLS


/**图像标定。*/
typedef struct _abcdk_torch_calibrate
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

} abcdk_torch_calibrate_t;

/**释放。*/
void abcdk_torch_calibrate_free_host(abcdk_torch_calibrate_t **ctx);

/**释放。*/
void abcdk_torch_calibrate_free_cuda(abcdk_torch_calibrate_t **ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_calibrate_free abcdk_torch_calibrate_free_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_calibrate_free abcdk_torch_calibrate_free_host
#endif //

/** 申请。*/
abcdk_torch_calibrate_t *abcdk_torch_calibrate_alloc_host();

/** 申请。*/
abcdk_torch_calibrate_t *abcdk_torch_calibrate_alloc_cuda();

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_calibrate_alloc abcdk_torch_calibrate_alloc_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_calibrate_alloc abcdk_torch_calibrate_alloc_host
#endif //

/**
 * 重置。
 * 
 * @note 默认的标定板尺寸是7行10列，格子尺寸是宽25毫米高25毫米。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_torch_calibrate_reset_host(abcdk_torch_calibrate_t *ctx, abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size);


/**
 * 重置。
 * 
 * @note 默认的标定板尺寸是7行10列，格子尺寸是宽25毫米高25毫米。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_torch_calibrate_reset_cuda(abcdk_torch_calibrate_t *ctx, abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size);


#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_calibrate_reset abcdk_torch_calibrate_reset_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_calibrate_reset abcdk_torch_calibrate_reset_host
#endif //

/**
 * 绑定图像。
 * 
 * @note 角点不存在或不足时，图像将被忽略。
 * 
 * @return 已绑定的数量。
 */
int abcdk_torch_calibrate_bind_host(abcdk_torch_calibrate_t *ctx, abcdk_torch_image_t *img);

/**
 * 绑定图像。
 * 
 * @note 角点不存在或不足时，图像将被忽略。
 * 
 * @return 已绑定的数量。
 */
int abcdk_torch_calibrate_bind_cuda(abcdk_torch_calibrate_t *ctx, abcdk_torch_image_t *img);


#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_calibrate_bind abcdk_torch_calibrate_bind_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_calibrate_bind abcdk_torch_calibrate_bind_host
#endif //


/**
 * 评估。
 * 
 * @note RMS值越少，质量越好。
 *
 * @return RMS。
 */
double abcdk_torch_calibrate_estimate_host(abcdk_torch_calibrate_t *ctx);

/**
 * 评估。
 *
 * @note RMS值越少，质量越好。
 *
 * @return RMS。
 */
double abcdk_torch_calibrate_estimate_cuda(abcdk_torch_calibrate_t *ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_calibrate_estimate abcdk_torch_calibrate_estimate_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_calibrate_estimate abcdk_torch_calibrate_estimate_host
#endif //

/**
 * 获取参数。
 *
 * @param [out] camera_matrix 内参矩阵。R,T.
 * @param [out] dist_coeffs 畸变系数。k1,k2,p1,p2,k3.
 * 
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_calibrate_getparam_host(abcdk_torch_calibrate_t *ctx, double camera_matrix[3][3], double dist_coeffs[5]);

/**
 * 获取参数。
 * 
 * @param [out] camera_matrix 内参矩阵。R,T.
 * @param [out] dist_coeffs 畸变系数。k1,k2,p1,p2,k3.
 * 
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_calibrate_getparam_cuda(abcdk_torch_calibrate_t *ctx, double camera_matrix[3][3], double dist_coeffs[5]);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_calibrate_getparam abcdk_torch_calibrate_getparam_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_calibrate_getparam abcdk_torch_calibrate_getparam_host
#endif //

/** 保存参数。*/
abcdk_object_t *abcdk_torch_calibrate_param_dump(const abcdk_torch_size_t *size, const double camera_matrix[3][3], const double dist_coeffs[5]);

/** 
 * 保存参数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_torch_calibrate_param_dump_file(const char *file, const abcdk_torch_size_t *size, const double camera_matrix[3][3], const double dist_coeffs[5]);

/** 
 * 加载参数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_torch_calibrate_param_load(abcdk_torch_size_t *size, double camera_matrix[3][3], double dist_coeffs[5], const char *data);

/** 
 * 加载参数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_torch_calibrate_param_load_file(abcdk_torch_size_t *size, double camera_matrix[3][3], double dist_coeffs[5], const char *file);


__END_DECLS

#endif // ABCDK_TORCH_CALIBRATE_H