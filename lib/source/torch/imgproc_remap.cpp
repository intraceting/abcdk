/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"
#include "abcdk/opencv/opencv.h"

__BEGIN_DECLS

#ifdef OPENCV_IMGPROC_HPP

static int _abcdk_torch_imgproc_remap_8u(int channels, int packed,
                                         uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h, const abcdk_torch_rect_t *dst_roi,
                                         const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h, const abcdk_torch_rect_t *src_roi,
                                         const float *xmap, size_t xmap_ws, const float *ymap, size_t ymap_ws,
                                         int inter_mode)
{
    abcdk_torch_size_t tmp_dst_size = {0}, tmp_src_size = {0};
    abcdk_torch_rect_t tmp_src_roi = {0};
    cv::Mat tmp_dst,tmp_src;
    cv::Mat tmp_xmap,tmp_ymap;

    assert(channels == 1 || channels == 3 || channels == 4);
    assert(dst != NULL && dst_w > 0 && dst_ws > 0 && dst_h > 0);
    assert(src != NULL && src_w > 0 && src_ws > 0 && src_h > 0);
    assert(xmap != NULL && xmap_ws > 0);
    assert(ymap != NULL && ymap_ws > 0);

    ABCDK_ASSERT(dst_roi == NULL && src_roi == NULL,"尚未支持感兴趣区域。");

    tmp_dst_size.width = dst_w;
    tmp_dst_size.height = dst_h;

    tmp_src_size.width = src_w;
    tmp_src_size.height = src_h;

    tmp_src_roi.x = (src_roi ? src_roi->x : 0);
    tmp_src_roi.y = (src_roi ? src_roi->y : 0);
    tmp_src_roi.width = (src_roi ? src_roi->width : src_w);
    tmp_src_roi.height = (src_roi ? src_roi->height : src_h);

    tmp_dst = cv::Mat(dst_h, dst_w, CV_8UC(channels), (void *)dst, dst_ws);
    tmp_src = cv::Mat(src_h, src_w, CV_8UC(channels), (void *)src, src_ws);
    tmp_xmap = cv::Mat(dst_h, dst_w, CV_32FC1, (void *)xmap, xmap_ws);
    tmp_ymap = cv::Mat(dst_h, dst_w, CV_32FC1, (void *)ymap, ymap_ws);

    cv::remap(tmp_src,tmp_dst,tmp_xmap,tmp_ymap,inter_mode);

    return 0;
}

int abcdk_torch_imgproc_remap_8u(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                 const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                 const abcdk_torch_image_t *xmap, const abcdk_torch_image_t *ymap,
                                 int inter_mode)
{
    int dst_depth;

    assert(dst != NULL && src != NULL);
    // assert(dst_roi != NULL && src_roi != NULL);
    assert(xmap != NULL && ymap != NULL);
    assert(dst->pixfmt == src->pixfmt);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_remap_8u(dst_depth, true,
                                         dst->data[0], dst->width, dst->stride[0], dst->height, dst_roi,
                                         src->data[0], src->width, src->stride[0], src->height, src_roi,
                                         (float *)xmap->data[0], xmap->stride[0], (float *)ymap->data[0], ymap->stride[0],
                                         inter_mode);
}

#else // OPENCV_IMGPROC_HPP

int abcdk_torch_imgproc_remap_8u(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                 const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                 const abcdk_torch_image_t *xmap, const abcdk_torch_image_t *ymap,
                                 int inter_mode)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return -1;
}

#endif // OPENCV_IMGPROC_HPP

__END_DECLS