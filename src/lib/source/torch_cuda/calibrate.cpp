/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/calibrate.h"
#include "abcdk/torch/nvidia.h"
#include "../torch/calibrate.hxx"

__BEGIN_DECLS

#if defined(__cuda_cuda_h__) && defined(OPENCV_CALIB3D_HPP)

void abcdk_torch_calibrate_free_cuda(abcdk_torch_calibrate_t **ctx)
{
    abcdk_torch_calibrate_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_CUDA);

    delete (abcdk::torch::calibrate *)ctx_p->private_ctx;
    abcdk_heap_free(ctx_p);
}

abcdk_torch_calibrate_t *abcdk_torch_calibrate_alloc_cuda()
{
    abcdk_torch_calibrate_t *ctx;

    ctx = (abcdk_torch_calibrate_t *)abcdk_heap_alloc(sizeof(abcdk_torch_calibrate_t));
    if (!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_CUDA;

    ctx->private_ctx = new abcdk::torch::calibrate();
    if (!ctx->private_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_torch_calibrate_free_cuda(&ctx);
    return NULL;
}

int abcdk_torch_calibrate_reset_cuda(abcdk_torch_calibrate_t *ctx, abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size)
{
    abcdk::torch::calibrate *ht_ctx_p;
    cv::Size tmp_board_size, tmp_grid_size;

    assert(ctx != NULL && board_size != NULL && grid_size != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);
    assert(board_size->width >= 2 && board_size->height >= 2);
    assert(grid_size->width > 0 && grid_size->height > 0);

    ht_ctx_p = (abcdk::torch::calibrate *)ctx->private_ctx;

    tmp_board_size.width = board_size->width;
    tmp_board_size.height = board_size->height;

    tmp_grid_size.width = grid_size->width;
    tmp_grid_size.height = grid_size->height;

    ht_ctx_p->Setup(tmp_board_size, tmp_grid_size);

    return 0;
}

int abcdk_torch_calibrate_bind_cuda(abcdk_torch_calibrate_t *ctx, abcdk_torch_image_t *img)
{
    abcdk::torch::calibrate *ht_ctx_p;
    cv::Mat tmp_img;
    int img_depth;
    int chk_count;

    assert(ctx != NULL && img != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);

    ht_ctx_p = (abcdk::torch::calibrate *)ctx->private_ctx;

    assert(img->tag == ABCDK_TORCH_TAG_CUDA);
    assert(img->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           img->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           img->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           img->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           img->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    img_depth = abcdk_torch_pixfmt_channels(img->pixfmt);

    /*创建cv::Mat对象。*/
    tmp_img.create(img->height, img->width, CV_8UC(img_depth));

    /*从设备复制到主机。*/
    abcdk_torch_memcpy_2d_cuda(tmp_img.data, tmp_img.step, 0, 0, 1,
                               img->data[0], img->stride[0], 0, 0, 0,
                               img->width * img_depth, img->height);

    chk_count = (int)ht_ctx_p->Bind(tmp_img);

    return chk_count;
}

double abcdk_torch_calibrate_estimate_cuda(abcdk_torch_calibrate_t *ctx)
{
    abcdk::torch::calibrate *ht_ctx_p;
    double chk_rms;

    assert(ctx != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);

    ht_ctx_p = (abcdk::torch::calibrate *)ctx->private_ctx;

    chk_rms = ht_ctx_p->Estimate();

    return chk_rms;
}

int abcdk_torch_calibrate_getparam_cuda(abcdk_torch_calibrate_t *ctx, double camera_matrix[3][3], double dist_coeffs[5])
{
    abcdk::torch::calibrate *ht_ctx_p;
    cv::Mat dst_camera_matrix, dst_dist_coeffs;

    assert(ctx != NULL && camera_matrix != NULL && dist_coeffs != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);

    ht_ctx_p = (abcdk::torch::calibrate *)ctx->private_ctx;

    ht_ctx_p->GetParam(dst_camera_matrix, dst_dist_coeffs);

    if(dst_camera_matrix.empty() || dst_dist_coeffs.empty())
        return -1;

    for (int y = 0; y < dst_camera_matrix.rows; y++)
    {
        for (int x = 0; x < dst_camera_matrix.cols; x++)
        {
            camera_matrix[y][x] = dst_camera_matrix.at<double>(y, x);
        }
    }

    for (int x = 0; x < dst_dist_coeffs.cols; x++)
    {
        dist_coeffs[x] = dst_dist_coeffs.at<double>(0, x);
    }

    return 0;
}

#else // #if defined(__cuda_cuda_h__) && defined(OPENCV_CALIB3D_HPP)

void abcdk_torch_calibrate_free_cuda(abcdk_torch_calibrate_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或OpenCV工具。"));
    return;
}

abcdk_torch_calibrate_t *abcdk_torch_calibrate_alloc_cuda()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或OpenCV工具。"));
    return NULL;
}

int abcdk_torch_calibrate_reset_cuda(abcdk_torch_calibrate_t *ctx, abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或OpenCV工具。"));
    return -1;
}

int abcdk_torch_calibrate_bind_cuda(abcdk_torch_calibrate_t *ctx, abcdk_torch_image_t *img)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或OpenCV工具。"));
    return 0;
}

double abcdk_torch_calibrate_estimate_cuda(abcdk_torch_calibrate_t *ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或OpenCV工具。"));
    return 1.0;
}

int  abcdk_torch_calibrate_getparam_cuda(abcdk_torch_calibrate_t *ctx, double camera_matrix[3][3], double dist_coeff[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或OpenCV工具。"));
    return -1;
}

#endif // #if defined(__cuda_cuda_h__) && defined(OPENCV_CALIB3D_HPP)

__END_DECLS
