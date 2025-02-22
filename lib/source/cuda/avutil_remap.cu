/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"


#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

int abcdk_cuda_avframe_remap(AVFrame *dst, const NppiRect *dst_roi,
                             const AVFrame *src, const NppiRect *src_roi,
                             const AVFrame *xmap, const AVFrame *ymap,
                             NppiInterpolationMode inter_mode)
{
    NppiSize tmp_dst_size = {0}, tmp_src_size = {0};
    NppiRect tmp_src_roi = {0};
    AVFrame *tmp_dst = NULL, *tmp_src = NULL, *tmp_xmap = NULL, *tmp_ymap = NULL;
    int dst_in_host, src_in_host, xmap_in_host, ymap_in_host;
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;
    int chk;

    assert(dst != NULL && src != NULL && xmap != NULL && ymap != NULL);

    assert(dst->format == src->format);
    assert(xmap->format == ymap->format);

    assert(dst->width == xmap->width && dst->height == xmap->height);
    assert(dst->width == ymap->width && dst->height == ymap->height);

    assert(dst->format == (int)AV_PIX_FMT_GRAY8 ||
           dst->format == (int)AV_PIX_FMT_RGB24 || dst->format == (int)AV_PIX_FMT_BGR24 ||
           dst->format == (int)AV_PIX_FMT_RGB32 || dst->format == (int)AV_PIX_FMT_BGR32);

    assert(xmap->format == (int)AV_PIX_FMT_GRAYF32 && ymap->format == (int)AV_PIX_FMT_GRAYF32);

    dst_in_host = (abcdk_cuda_avframe_memory_type(dst) != CU_MEMORYTYPE_DEVICE);
    src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);
    xmap_in_host = (abcdk_cuda_avframe_memory_type(xmap) != CU_MEMORYTYPE_DEVICE);
    ymap_in_host = (abcdk_cuda_avframe_memory_type(ymap) != CU_MEMORYTYPE_DEVICE);

    if (xmap_in_host)
    {
        tmp_xmap = abcdk_cuda_avframe_clone(0, xmap);
        if (!tmp_xmap)
            return -1;

        chk = abcdk_cuda_avframe_remap(dst, dst_roi, src, src_roi, tmp_xmap, ymap, inter_mode);
        av_frame_free(&tmp_xmap);

        return chk;
    }

    if (ymap_in_host)
    {
        tmp_ymap = abcdk_cuda_avframe_clone(0, ymap);
        if (!tmp_ymap)
            return -1;

        chk = abcdk_cuda_avframe_remap(dst, dst_roi, src, src_roi, xmap, tmp_ymap, inter_mode);
        av_frame_free(&tmp_ymap);

        return chk;
    }

    if (src_in_host)
    {
        tmp_src = abcdk_cuda_avframe_clone(0, src);
        if (!tmp_src)
            return -1;

        chk = abcdk_cuda_avframe_remap(dst, dst_roi, tmp_src, src_roi, xmap, ymap, inter_mode);
        av_frame_free(&tmp_src);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if (dst_in_host)
    {
        tmp_dst = abcdk_cuda_avframe_clone(0, dst);
        if (!tmp_dst)
            return -1;

        chk = abcdk_cuda_avframe_remap(tmp_dst, dst_roi, src, src_roi, xmap, ymap, inter_mode);
        if (chk == 0)
            abcdk_cuda_avframe_copy(dst, tmp_dst);
        av_frame_free(&tmp_dst);

        return chk;
    }

    tmp_dst_size.width = dst->width;
    tmp_dst_size.height = dst->height;

    tmp_src_size.width = src->width;
    tmp_src_size.height = src->height;

    tmp_src_roi.x = (src_roi ? src_roi->x : 0);
    tmp_src_roi.y = (src_roi ? src_roi->y : 0);
    tmp_src_roi.width = (src_roi ? src_roi->width : src->width);
    tmp_src_roi.height = (src_roi ? src_roi->height : src->height);

    if (dst->format == (int)AV_PIX_FMT_GRAY8)
    {
        npp_chk = nppiRemap_8u_C1R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                   (Npp32f *)xmap->data[0], xmap->linesize[0],
                                   (Npp32f *)ymap->data[0], ymap->linesize[0],
                                   dst->data[0], dst->linesize[0], tmp_dst_size,
                                   inter_mode);
    }
    else if (dst->format == (int)AV_PIX_FMT_RGB24 || dst->format == (int)AV_PIX_FMT_BGR24)
    {
        npp_chk = nppiRemap_8u_C3R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                   (Npp32f *)xmap->data[0], xmap->linesize[0],
                                   (Npp32f *)ymap->data[0], ymap->linesize[0],
                                   dst->data[0], dst->linesize[0], tmp_dst_size,
                                   inter_mode);
    }
    else if (dst->format == (int)AV_PIX_FMT_RGB32 || dst->format == (int)AV_PIX_FMT_BGR32)
    {
        npp_chk = nppiRemap_8u_C4R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                   (Npp32f *)xmap->data[0], xmap->linesize[0],
                                   (Npp32f *)ymap->data[0], ymap->linesize[0],
                                   dst->data[0], dst->linesize[0], tmp_dst_size,
                                   inter_mode);
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__