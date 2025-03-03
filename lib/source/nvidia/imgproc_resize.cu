/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

int abcdk_cuda_imgproc_resize_8u(int channels, int packed,
                                 uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h, const abcdk_torch_rect_t *dst_roi,
                                 const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h, const abcdk_torch_rect_t *src_roi,
                                 int keep_aspect_ratio, NppiInterpolationMode inter_mode)
{
    NppiSize tmp_src_size = {0};
    NppiRect tmp_dst_roi = {0}, tmp_src_roi = {0};
    abcdk_resize_t tmp_param = {0};
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

    abcdk_resize_ratio_2d(&tmp_param,tmp_src_roi.width,tmp_src_roi.height,tmp_dst_roi.width,tmp_dst_roi.height,keep_aspect_ratio);

    if (channels == 1)
    {
        npp_chk = nppiResizeSqrPixel_8u_C1R(src, tmp_src_size, src_ws, tmp_src_roi,
                                            dst, dst_ws, tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }
    else if (channels == 3)
    {
        npp_chk = nppiResizeSqrPixel_8u_C3R(src, tmp_src_size, src_ws, tmp_src_roi,
                                            dst, dst_ws, tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }
    else if (channels == 4)
    {
        npp_chk = nppiResizeSqrPixel_8u_C4R(src, tmp_src_size, src_ws, tmp_src_roi,
                                            dst, dst_ws, tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}


#else // __cuda_cuda_h__

int abcdk_cuda_imgproc_resize_8u(int channels, int packed,
                                 uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h, const abcdk_torch_rect_t *dst_roi,
                                 const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h, const abcdk_torch_rect_t *src_roi,
                                 int keep_aspect_ratio, NppiInterpolationMode inter_mode)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

#endif //__cuda_cuda_h__

__END_DECLS
