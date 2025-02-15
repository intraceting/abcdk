/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

static void _abcdk_cuda_avframe_resize_free_tmp(AVFrame *dst, const AVFrame *src, AVFrame *tmp_dst, AVFrame *tmp_src)
{
    if (tmp_dst != dst)
        av_frame_free(&tmp_dst);

    if (tmp_src != src)
        av_frame_free(&tmp_src);
}

int abcdk_cuda_avframe_resize(AVFrame *dst, const NppiRect *dst_roi, int dst_in_host,
                              const AVFrame *src, const NppiRect *src_roi, int src_in_host,
                              int keep_aspect_ratio, NppiInterpolationMode inter_mode)
{

    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    NppiSize tmp_src_size = {0};
    NppiRect tmp_dst_roi = {0}, tmp_src_roi = {0};
    abcdk_resize_t tmp_param = {0};
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;

    assert(dst != NULL && src != NULL);
    assert(dst->format ==  src->format);
    assert(src->format == (int)AV_PIX_FMT_GRAY8 ||
           src->format == (int)AV_PIX_FMT_RGB24 || src->format == (int)AV_PIX_FMT_BGR24 ||
           src->format == (int)AV_PIX_FMT_RGB32 || src->format == (int)AV_PIX_FMT_BGR32);

    tmp_dst = (dst_in_host ? abcdk_cuda_avframe_clone(0, dst, 1) : dst);
    tmp_src = (src_in_host ? abcdk_cuda_avframe_clone(0, src, 1) : (AVFrame *)src);

    if (!tmp_dst || !tmp_src)
    {
        _abcdk_cuda_avframe_resize_free_tmp(dst, src, tmp_dst, tmp_src);
        return -1;
    }

    tmp_dst_roi.x = (dst_roi ? dst_roi->x : 0);
    tmp_dst_roi.y = (dst_roi ? dst_roi->y : 0);
    tmp_dst_roi.width = (dst_roi ? dst_roi->width : tmp_dst->width);
    tmp_dst_roi.height = (dst_roi ? dst_roi->height : tmp_dst->height);

    tmp_src_size.width = tmp_src->width;
    tmp_src_size.height = tmp_src->height;

    tmp_src_roi.x = (src_roi ? src_roi->x : 0);
    tmp_src_roi.y = (src_roi ? src_roi->y : 0);
    tmp_src_roi.width = (src_roi ? src_roi->width : tmp_src->width);
    tmp_src_roi.height = (src_roi ? src_roi->height : tmp_src->height);

    abcdk_resize_ratio_2d(&tmp_param,tmp_src_roi.width,tmp_src_roi.height,tmp_dst_roi.width,tmp_dst_roi.height,keep_aspect_ratio);

    if (src->format == (int)AV_PIX_FMT_GRAY8)
    {
        npp_chk = nppiResizeSqrPixel_8u_C1R(tmp_src->data[0], tmp_src_size, tmp_src->linesize[0], tmp_src_roi,
                                            tmp_dst->data[0], tmp_dst->linesize[0], tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }
    else if (src->format == (int)AV_PIX_FMT_RGB24 || src->format == (int)AV_PIX_FMT_BGR24)
    {
        npp_chk = nppiResizeSqrPixel_8u_C3R(tmp_src->data[0], tmp_src_size, tmp_src->linesize[0], tmp_src_roi,
                                            tmp_dst->data[0], tmp_dst->linesize[0], tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }
    else if (src->format == (int)AV_PIX_FMT_RGB32 || src->format == (int)AV_PIX_FMT_BGR32)
    {
        npp_chk = nppiResizeSqrPixel_8u_C4R(tmp_src->data[0], tmp_src_size, tmp_src->linesize[0], tmp_src_roi,
                                            tmp_dst->data[0], tmp_dst->linesize[0], tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }

    /*按需复制。*/
    if (npp_chk == NPP_SUCCESS && dst != tmp_dst)
        abcdk_cuda_avframe_copy(dst, 1, tmp_dst, 0);

    _abcdk_cuda_avframe_resize_free_tmp(dst, src, tmp_dst, tmp_src);

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

#endif //__cuda_cuda_h__