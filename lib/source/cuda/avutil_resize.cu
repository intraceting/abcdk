/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"


#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

int abcdk_cuda_avframe_resize(AVFrame *dst, const NppiRect *dst_roi,
                              const AVFrame *src, const NppiRect *src_roi,
                              int keep_aspect_ratio, NppiInterpolationMode inter_mode)
{
    NppiSize tmp_src_size = {0};
    NppiRect tmp_dst_roi = {0}, tmp_src_roi = {0};
    abcdk_resize_t tmp_param = {0};
    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    int dst_in_host, src_in_host;
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->format ==  src->format);
    assert(src->format == (int)AV_PIX_FMT_GRAY8 ||
           src->format == (int)AV_PIX_FMT_RGB24 || src->format == (int)AV_PIX_FMT_BGR24 ||
           src->format == (int)AV_PIX_FMT_RGB32 || src->format == (int)AV_PIX_FMT_BGR32);

    dst_in_host = (abcdk_cuda_avframe_memory_type(dst) != CU_MEMORYTYPE_DEVICE);
    src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);

    if (src_in_host)
    {
        tmp_src = abcdk_cuda_avframe_clone(0, src);
        if (!tmp_src)
            return -1;

        chk = abcdk_cuda_avframe_resize(dst, dst_roi, tmp_src, src_roi, keep_aspect_ratio, inter_mode);
        av_frame_free(&tmp_src);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if (dst_in_host)
    {
        tmp_dst = abcdk_cuda_avframe_clone(0, dst);
        if (!tmp_dst)
            return -1;

        chk = abcdk_cuda_avframe_resize(tmp_dst, dst_roi, src, src_roi, keep_aspect_ratio, inter_mode);
        if (chk == 0)
            abcdk_cuda_avframe_copy(dst, tmp_dst);
        av_frame_free(&tmp_dst);

        return chk;
    }

    tmp_dst_roi.x = (dst_roi ? dst_roi->x : 0);
    tmp_dst_roi.y = (dst_roi ? dst_roi->y : 0);
    tmp_dst_roi.width = (dst_roi ? dst_roi->width : dst->width);
    tmp_dst_roi.height = (dst_roi ? dst_roi->height : dst->height);

    tmp_src_size.width = src->width;
    tmp_src_size.height = src->height;

    tmp_src_roi.x = (src_roi ? src_roi->x : 0);
    tmp_src_roi.y = (src_roi ? src_roi->y : 0);
    tmp_src_roi.width = (src_roi ? src_roi->width : src->width);
    tmp_src_roi.height = (src_roi ? src_roi->height : src->height);

    abcdk_resize_ratio_2d(&tmp_param,tmp_src_roi.width,tmp_src_roi.height,tmp_dst_roi.width,tmp_dst_roi.height,keep_aspect_ratio);

    if (src->format == (int)AV_PIX_FMT_GRAY8)
    {
        npp_chk = nppiResizeSqrPixel_8u_C1R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                            dst->data[0], dst->linesize[0], tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }
    else if (src->format == (int)AV_PIX_FMT_RGB24 || src->format == (int)AV_PIX_FMT_BGR24)
    {
        npp_chk = nppiResizeSqrPixel_8u_C3R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                            dst->data[0], dst->linesize[0], tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }
    else if (src->format == (int)AV_PIX_FMT_RGB32 || src->format == (int)AV_PIX_FMT_BGR32)
    {
        npp_chk = nppiResizeSqrPixel_8u_C4R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                            dst->data[0], dst->linesize[0], tmp_dst_roi,
                                            tmp_param.x_factor, tmp_param.y_factor, tmp_param.x_shift, tmp_param.y_shift, (int)inter_mode);
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__