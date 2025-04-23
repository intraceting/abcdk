/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/calibrate.h"
#include "../torch/calibrate.hxx"

__BEGIN_DECLS

double abcdk_torch_calibrate_estimate_2d_host(abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size, int count, abcdk_torch_image_t *img[], float camera_matrix[3][3], float dist_coeffs[5])
{
    cv::Size tmp_board_size, tmp_grid_size;
    std::vector<cv::Mat> tmp_img;
    cv::Mat dst_camera_matrix, dst_dist_coeffs;
    double chk_rms;

    assert(board_size != NULL && grid_size != NULL && count >= 2 && img != NULL && camera_matrix != NULL && dist_coeffs != NULL);
    assert(board_size->width > 0 && board_size->height >0);
    assert(grid_size->width > 0 && grid_size->height >0);

    tmp_board_size.width = board_size->width;
    tmp_board_size.height = board_size->height;

    tmp_grid_size.width = grid_size->width;
    tmp_grid_size.height = grid_size->height;

    tmp_img.resize(count);

    for (int i = 0; i < count; i++)
    {
        auto &img_p = img[i];

        assert(img_p != NULL);
        assert(img_p->tag == ABCDK_TORCH_TAG_HOST);
        assert(img_p->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
               img_p->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
               img_p->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
               img_p->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
               img_p->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

        int img_depth = abcdk_torch_pixfmt_channels(img_p->pixfmt);

        /*用已存在数据构造cv::Mat对象。*/
        tmp_img[i] = cv::Mat(img_p->height, img_p->width, CV_8UC(img_depth), (void *)img_p->data[0], img_p->stride[0]);
    }

    chk_rms = abcdk::torch::calibrate::Estimate(tmp_board_size, tmp_grid_size, tmp_img, dst_camera_matrix, dst_dist_coeffs);

    for (int y = 0; y < dst_camera_matrix.rows; y++)
    {
        for (int x = 0; x < dst_camera_matrix.cols; x++)
        {
            camera_matrix[y][x] = dst_camera_matrix.at<float>(y, x);
        }
    }

    for (int x = 0; x < dst_dist_coeffs.cols; x++)
    {
        dist_coeffs[x] = dst_dist_coeffs.at<float>(0, x);
    }

    return chk_rms;
}

__END_DECLS
