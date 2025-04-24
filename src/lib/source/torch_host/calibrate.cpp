/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/calibrate.h"
#include "../torch/calibrate.hxx"

__BEGIN_DECLS

#ifdef OPENCV_CALIB3D_HPP

void abcdk_torch_calibrate_free_host(abcdk_torch_calibrate_t **ctx)
{
    abcdk_torch_calibrate_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    delete (abcdk::torch::calibrate *)ctx_p->private_ctx;
    abcdk_heap_free(ctx_p);
}

abcdk_torch_calibrate_t *abcdk_torch_calibrate_alloc_host()
{
    abcdk_torch_calibrate_t *ctx;

    ctx = (abcdk_torch_calibrate_t *)abcdk_heap_alloc(sizeof(abcdk_torch_calibrate_t));
    if (!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_HOST;

    ctx->private_ctx = new abcdk::torch::calibrate();
    if (!ctx->private_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_torch_calibrate_free_host(&ctx);
    return NULL;
}

int abcdk_torch_calibrate_reset_host(abcdk_torch_calibrate_t *ctx, abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size)
{
    abcdk::torch::calibrate *ht_ctx_p;
    cv::Size tmp_board_size, tmp_grid_size;

    assert(ctx != NULL && board_size != NULL && grid_size != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);
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

int abcdk_torch_calibrate_bind_host(abcdk_torch_calibrate_t *ctx, abcdk_torch_image_t *img)
{
    abcdk::torch::calibrate *ht_ctx_p;
    cv::Mat tmp_img;
    int img_depth;
    int chk_count;

    assert(ctx != NULL && img != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk::torch::calibrate *)ctx->private_ctx;

    assert(img->tag == ABCDK_TORCH_TAG_HOST);
    assert(img->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           img->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           img->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           img->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           img->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    img_depth = abcdk_torch_pixfmt_channels(img->pixfmt);

    /*用已存在数据构造cv::Mat对象。*/
    tmp_img = cv::Mat(img->height, img->width, CV_8UC(img_depth), (void *)img->data[0], img->stride[0]);

    chk_count = (int)ht_ctx_p->Bind(tmp_img);

    return chk_count;
}

double abcdk_torch_calibrate_estimate_host(abcdk_torch_calibrate_t *ctx)
{
    abcdk::torch::calibrate *ht_ctx_p;
    double chk_rms;

    assert(ctx != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk::torch::calibrate *)ctx->private_ctx;

    chk_rms = ht_ctx_p->Estimate();

    return chk_rms;
}

int abcdk_torch_calibrate_getparam_host(abcdk_torch_calibrate_t *ctx, double camera_matrix[3][3], double dist_coeffs[5])
{
    abcdk::torch::calibrate *ht_ctx_p;
    cv::Mat dst_camera_matrix, dst_dist_coeffs;

    assert(ctx != NULL && camera_matrix != NULL && dist_coeffs != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk::torch::calibrate *)ctx->private_ctx;

    ht_ctx_p->GetParam(dst_camera_matrix, dst_dist_coeffs);

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

abcdk_object_t *abcdk_torch_calibrate_param_dump(const double camera_matrix[3][3], const double dist_coeffs[5])
{
    cv::Mat tmp_camera_matrix, tmp_dist_coeffs;
    std::string metadata;

    assert(camera_matrix != NULL && dist_coeffs != NULL);

    tmp_camera_matrix = cv::Mat(3, 3, CV_64FC1, (void *)camera_matrix, 3 * sizeof(double));
    tmp_dist_coeffs = cv::Mat(1, 5, CV_64FC1, (void *)dist_coeffs, 5 * sizeof(double));

    cv::FileStorage fd("{}", cv::FileStorage::MEMORY | cv::FileStorage::WRITE | cv::FileStorage::FORMAT_XML);
    if (!fd.isOpened())
        return NULL;

    cv::write(fd, "camera_matrix",tmp_camera_matrix);
    cv::write(fd, "dist_coeffs",tmp_dist_coeffs);

    metadata = fd.releaseAndGetString();
    if(metadata.length() <= 0)
        return NULL;

    return abcdk_object_copyfrom(metadata.data(),metadata.length());
}

int abcdk_torch_calibrate_param_dump_file(const char *file, const double camera_matrix[3][3], const double dist_coeffs[5])
{
    abcdk_object_t *metadata;
    int chk;

    assert(file != NULL && camera_matrix != NULL && dist_coeffs != NULL);

    abcdk_mkdir(file, 0755);

    if (access(file, F_OK) == 0)
    {
        chk = truncate(file, 0);
        if (chk != 0)
            return -1;
    }

    metadata = abcdk_torch_calibrate_param_dump(camera_matrix, dist_coeffs);
    if (!metadata)
        return -1;

    chk = abcdk_save(file, metadata->pptrs[0], metadata->sizes[0], 0);
    abcdk_object_unref(&metadata);

    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_torch_calibrate_param_load(double camera_matrix[3][3], double dist_coeffs[5], const char *data)
{
    cv::Mat tmp_camera_matrix, tmp_dist_coeffs;

    assert(camera_matrix != NULL && dist_coeffs != NULL && data != NULL);

    tmp_camera_matrix = cv::Mat(3, 3, CV_64FC1, (void *)camera_matrix, 3 * sizeof(double));
    tmp_dist_coeffs = cv::Mat(1, 5, CV_64FC1, (void *)dist_coeffs, 5 * sizeof(double));

    cv::FileStorage fd(data, cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_XML);
    if (!fd.isOpened())
        return -1;

    cv::FileNode camera_matrix_node = fd["camera_matrix"];
    if (camera_matrix_node.empty())
        return -1;

    cv::FileNode dist_coeffs_node = fd["dist_coeffs"];
    if (dist_coeffs_node.empty())
        return -1;

    camera_matrix_node.mat().copyTo(tmp_camera_matrix);
    dist_coeffs_node.mat().copyTo(tmp_dist_coeffs);

    return 0;
}

int abcdk_torch_calibrate_param_load_file(double camera_matrix[3][3], double dist_coeffs[5], const char *file)
{
    abcdk_object_t *metadata;
    int chk;

    assert(camera_matrix != NULL && dist_coeffs != NULL && file != NULL);

    metadata = abcdk_object_copyfrom_file(file);
    if(!metadata)
        return -1;

    chk = abcdk_torch_calibrate_param_load(camera_matrix,dist_coeffs,metadata->pstrs[0]);
    abcdk_object_unref(&metadata);

    if (chk != 0)
        return -1;

    return 0;
}


#else // OPENCV_CALIB3D_HPP

void abcdk_torch_calibrate_free_host(abcdk_torch_calibrate_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return;
}

abcdk_torch_calibrate_t *abcdk_torch_calibrate_alloc_host()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

int abcdk_torch_calibrate_reset_host(abcdk_torch_calibrate_t *ctx, abcdk_torch_size_t *board_size, abcdk_torch_size_t *grid_size)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

int abcdk_torch_calibrate_bind_host(abcdk_torch_calibrate_t *ctx, abcdk_torch_image_t *img)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return 0;
}

double abcdk_torch_calibrate_estimate_host(abcdk_torch_calibrate_t *ctx, double camera_matrix[3][3], double dist_coeffs[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return 1.0;
}

int abcdk_torch_calibrate_getparam_host(abcdk_torch_calibrate_t *ctx, double camera_matrix[3][3], double dist_coeffs[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

abcdk_object_t *abcdk_torch_calibrate_param_dump(const double camera_matrix[3][3], const double dist_coeffs[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

int abcdk_torch_calibrate_param_dump_file(const char *file, const double camera_matrix[3][3], const double dist_coeffs[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

int abcdk_torch_calibrate_param_load(double camera_matrix[3][3], double dist_coeffs[5], const char *data)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

int abcdk_torch_calibrate_param_load_file(double camera_matrix[3][3], double dist_coeffs[5], const char *file)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // OPENCV_CALIB3D_HPP

__END_DECLS
