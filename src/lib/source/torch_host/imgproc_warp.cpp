/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/opencv.h"

__BEGIN_DECLS

#ifdef OPENCV_IMGPROC_HPP

static int _abcdk_torch_imgproc_warp_8u_host(int channels, int packed,
                                             uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h, const abcdk_torch_rect_t *dst_roi, const abcdk_torch_point_t dst_quad[4],
                                             const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h, const abcdk_torch_rect_t *src_roi, const abcdk_torch_point_t src_quad[4],
                                             int warp_mode, int inter_mode)
{

    abcdk_torch_size_t tmp_src_size = {0};
    abcdk_torch_rect_t tmp_dst_roi = {0}, tmp_src_roi = {0};
    std::vector<cv::Point2f> tmp_dst_quad(4), tmp_src_quad(4);
    cv::Mat tmp_dst, tmp_src;
    cv::Mat m;

    assert(channels == 1 || channels == 3 || channels == 4);
    assert(dst != NULL && dst_w > 0 && dst_ws > 0 && dst_h > 0);
    assert(src != NULL && src_w > 0 && src_ws > 0 && src_h > 0);

    ABCDK_ASSERT(dst_roi == NULL && src_roi == NULL, TT("尚未支持感兴趣区域。"));

    tmp_dst_roi.x = (dst_roi ? dst_roi->x : 0);
    tmp_dst_roi.y = (dst_roi ? dst_roi->y : 0);
    tmp_dst_roi.width = (dst_roi ? dst_roi->width : dst_w);
    tmp_dst_roi.height = (dst_roi ? dst_roi->height : dst_h);

    tmp_src_size.width = src_w;
    tmp_src_size.height = src_h;

    tmp_src_roi.x = (src_roi ? src_roi->x : 0);
    tmp_src_roi.y = (src_roi ? src_roi->y : 0);
    tmp_src_roi.width = (src_roi ? src_roi->width : src_w);
    tmp_src_roi.height = (src_roi ? src_roi->height : src_h);

    if (dst_quad)
    {
        for (int i = 0; i < 4; i++)
        {
            tmp_dst_quad[i].x = dst_quad[i].x;
            tmp_dst_quad[i].y = dst_quad[i].y;
        }
    }
    else
    {
        tmp_dst_quad[0].x = 0;
        tmp_dst_quad[0].y = 0;
        tmp_dst_quad[1].x = dst_w - 1;
        tmp_dst_quad[1].y = 0;
        tmp_dst_quad[2].x = dst_w - 1;
        tmp_dst_quad[2].y = dst_h - 1;
        tmp_dst_quad[3].x = 0;
        tmp_dst_quad[3].y = dst_h - 1;
    }

    if (src_quad)
    {
        for (int i = 0; i < 4; i++)
        {
            tmp_src_quad[i].x = src_quad[i].x;
            tmp_src_quad[i].y = src_quad[i].y;
        }
    }
    else
    {
        tmp_src_quad[0].x = 0;
        tmp_src_quad[0].y = 0;
        tmp_src_quad[1].x = src_w - 1;
        tmp_src_quad[1].y = 0;
        tmp_src_quad[2].x = src_w - 1;
        tmp_src_quad[2].y = src_h - 1;
        tmp_src_quad[3].x = 0;
        tmp_src_quad[3].y = src_h - 1;
    }

    tmp_dst = cv::Mat(dst_h, dst_w, CV_8UC(channels), (void *)dst, dst_ws);
    tmp_src = cv::Mat(src_h, src_w, CV_8UC(channels), (void *)src, src_ws);

    m = cv::findHomography(tmp_src_quad, tmp_dst_quad, 0);

    if (warp_mode == 1)
    {
        cv::warpPerspective(tmp_src, tmp_dst, m, cv::Size(dst_w, dst_h), inter_mode);
    }
    else if (warp_mode == 2)
    {
        cv::warpAffine(tmp_src, tmp_dst, m, cv::Size(dst_w, dst_h), inter_mode);
    }

    return 0;
}

int abcdk_torch_imgproc_warp_host(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi, const abcdk_torch_point_t dst_quad[4],
                                  const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi, const abcdk_torch_point_t src_quad[4],
                                  int warp_mode, int inter_mode)
{
    int dst_depth;

    assert(dst != NULL && src != NULL);
    // assert(dst_roi != NULL && src_roi != NULL);
    // assert(dst_quad != NULL && src_quad != NULL);
    assert(dst->pixfmt == src->pixfmt);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_HOST);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_warp_8u_host(dst_depth, true,
                                             dst->data[0], dst->width, dst->stride[0], dst->height, dst_roi, dst_quad,
                                             src->data[0], src->width, src->stride[0], src->height, src_roi, src_quad,
                                             warp_mode, inter_mode);
}

#else // OPENCV_IMGPROC_HPP

int abcdk_torch_imgproc_warp_host(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi, const abcdk_torch_point_t dst_quad[4],
                                  const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi, const abcdk_torch_point_t src_quad[4],
                                  int warp_mode, int inter_mode)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // OPENCV_IMGPROC_HPP

__END_DECLS