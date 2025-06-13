/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"
#include "inter_mode.hxx"

#ifdef __cuda_cuda_h__

static int _abcdk_torch_imgproc_resize_cuda(int channels, int packed, int pixfmt,
                                            uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h, const abcdk_torch_rect_t *dst_roi,
                                            const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h, const abcdk_torch_rect_t *src_roi,
                                            int keep_aspect_ratio, int inter_mode)
{
    NppiSize tmp_src_size = {0};
    NppiRect tmp_dst_roi = {0}, tmp_src_roi = {0};
    abcdk_resize_scale_t tmp_param = {0};
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;

    assert(channels == 1 || channels == 3 || channels == 4);
    assert(dst != NULL && dst_w > 0 && dst_ws > 0 && dst_h > 0);
    assert(src != NULL && src_w > 0 && src_ws > 0 && src_h > 0);

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

    abcdk_resize_ratio_2d(&tmp_param, tmp_src_roi.width, tmp_src_roi.height, tmp_dst_roi.width, tmp_dst_roi.height, keep_aspect_ratio);

    if (channels == 1)
    {
        if (pixfmt == ABCDK_TORCH_PIXFMT_GRAY8)
        {
            npp_chk = nppiResizeSqrPixel_8u_C1R(src, tmp_src_size, src_ws, tmp_src_roi,
                                                dst, dst_ws, tmp_dst_roi,
                                                tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift,
                                                abcdk::torch_cuda::inter_mode::convert2nppi(inter_mode));
        }
        else if (pixfmt == ABCDK_TORCH_PIXFMT_GRAYF32)
        {
            npp_chk = nppiResizeSqrPixel_32f_C1R((Npp32f *)src, tmp_src_size, src_ws, tmp_src_roi,
                                                 (Npp32f *)dst, dst_ws, tmp_dst_roi,
                                                 tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift,
                                                 abcdk::torch_cuda::inter_mode::convert2nppi(inter_mode));
        }

    }
    else if (channels == 3)
    {
        if (pixfmt == ABCDK_TORCH_PIXFMT_RGB24 || pixfmt == ABCDK_TORCH_PIXFMT_BGR24)
        {
            npp_chk = nppiResizeSqrPixel_8u_C3R(src, tmp_src_size, src_ws, tmp_src_roi,
                                                dst, dst_ws, tmp_dst_roi,
                                                tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift,
                                                abcdk::torch_cuda::inter_mode::convert2nppi(inter_mode));
        }
        else if (pixfmt == ABCDK_TORCH_PIXFMT_GRAYF32)
        {
            npp_chk = nppiResizeSqrPixel_32f_C3R((Npp32f *)src, tmp_src_size, src_ws, tmp_src_roi,
                                                 (Npp32f *)dst, dst_ws, tmp_dst_roi,
                                                 tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift,
                                                 abcdk::torch_cuda::inter_mode::convert2nppi(inter_mode));
        }
    }
    else if (channels == 4)
    {
        if (pixfmt == ABCDK_TORCH_PIXFMT_RGB32 || pixfmt == ABCDK_TORCH_PIXFMT_BGR32)
        {
            npp_chk = nppiResizeSqrPixel_8u_C4R(src, tmp_src_size, src_ws, tmp_src_roi,
                                                dst, dst_ws, tmp_dst_roi,
                                                tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift,
                                                abcdk::torch_cuda::inter_mode::convert2nppi(inter_mode));
        }
        else if (pixfmt == ABCDK_TORCH_PIXFMT_GRAYF32)
        {
            npp_chk = nppiResizeSqrPixel_32f_C4R((Npp32f *)src, tmp_src_size, src_ws, tmp_src_roi,
                                                 (Npp32f *)dst, dst_ws, tmp_dst_roi,
                                                 tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift,
                                                 abcdk::torch_cuda::inter_mode::convert2nppi(inter_mode));
        }
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_resize_cuda(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                    const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                    int keep_aspect_ratio, int inter_mode)
{
    int dst_depth;

    assert(dst != NULL && src != NULL);
    // assert(dst_roi != NULL && src_roi != NULL);
    assert(dst->pixfmt == src->pixfmt);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAYF32);

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->tag == ABCDK_TORCH_TAG_CUDA);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_resize_cuda(dst_depth, true, dst->pixfmt,
                                            dst->data[0], dst->width, dst->stride[0], dst->height, dst_roi,
                                            src->data[0], src->width, src->stride[0], src->height, src_roi,
                                            keep_aspect_ratio, inter_mode);
}

__END_DECLS


#endif //__cuda_cuda_h__
