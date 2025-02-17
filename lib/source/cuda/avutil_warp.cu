/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"

#ifdef __cuda_cuda_h__

int abcdk_cuda_avframe_warp(AVFrame *dst, const NppiRect *dst_roi, const NppiPoint dst_quad[4],
                            const AVFrame *src, const NppiRect *src_roi, const NppiPoint src_quad[4],
                            int warp_mode, NppiInterpolationMode inter_mode)
{

    NppiSize tmp_src_size = {0};
    NppiRect tmp_dst_roi = {0}, tmp_src_roi = {0};
    double tmp_dst_quad[4][2], tmp_src_quad[4][2];
    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    int dst_in_host, src_in_host;
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->format == src->format);
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

        chk = abcdk_cuda_avframe_warp(dst, dst_roi, dst_quad, tmp_src, src_roi, src_quad, warp_mode, inter_mode);
        av_frame_free(&tmp_src);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if (dst_in_host)
    {
        tmp_dst = abcdk_cuda_avframe_clone(0, dst);
        if (!tmp_dst)
            return -1;

        chk = abcdk_cuda_avframe_warp(tmp_dst, dst_roi, dst_quad, src, src_roi, src_quad, warp_mode, inter_mode);
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

    if (dst_quad)
    {
        for (int i = 0; i < 4; i++)
        {
            tmp_dst_quad[i][0] = dst_quad[i].x;
            tmp_dst_quad[i][1] = dst_quad[i].y;
        }
    }
    else
    {
        tmp_dst_quad[0][0] = 0;
        tmp_dst_quad[0][1] = 0;
        tmp_dst_quad[1][0] = dst->width - 1;
        tmp_dst_quad[1][1] = 0;
        tmp_dst_quad[2][0] = dst->width - 1;
        tmp_dst_quad[2][1] = dst->height - 1;
        tmp_dst_quad[3][0] = 0;
        tmp_dst_quad[3][1] = dst->height - 1;
    }

    if (src_quad)
    {
        for (int i = 0; i < 4; i++)
        {
            tmp_src_quad[i][0] = src_quad[i].x;
            tmp_src_quad[i][1] = src_quad[i].y;
        }
    }
    else
    {
        tmp_src_quad[0][0] = 0;
        tmp_src_quad[0][1] = 0;
        tmp_src_quad[1][0] = src->width - 1;
        tmp_src_quad[1][1] = 0;
        tmp_src_quad[2][0] = src->width - 1;
        tmp_src_quad[2][1] = src->height - 1;
        tmp_src_quad[3][0] = 0;
        tmp_src_quad[3][1] = src->height - 1;
    }

    if (warp_mode == 1)
    {
        if (src->format == (int)AV_PIX_FMT_GRAY8)
        {

            npp_chk = nppiWarpPerspectiveQuad_8u_C1R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi, tmp_src_quad,
                                                     dst->data[0], dst->linesize[0], tmp_dst_roi, tmp_dst_quad,
                                                     (int)inter_mode);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB24 || src->format == (int)AV_PIX_FMT_BGR24)
        {
            npp_chk = nppiWarpPerspectiveQuad_8u_C3R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi, tmp_src_quad,
                                                     dst->data[0], dst->linesize[0], tmp_dst_roi, tmp_dst_quad,
                                                     (int)inter_mode);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB32 || src->format == (int)AV_PIX_FMT_BGR32)
        {

            npp_chk = nppiWarpPerspectiveQuad_8u_C4R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi, tmp_src_quad,
                                                     dst->data[0], dst->linesize[0], tmp_dst_roi, tmp_dst_quad,
                                                     (int)inter_mode);
        }
    }
    else if (warp_mode == 2)
    {
        if (src->format == (int)AV_PIX_FMT_GRAY8)
        {

            npp_chk = nppiWarpAffineQuad_8u_C1R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi, tmp_src_quad,
                                                dst->data[0], dst->linesize[0], tmp_dst_roi, tmp_dst_quad,
                                                (int)inter_mode);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB24 || src->format == (int)AV_PIX_FMT_BGR24)
        {
            npp_chk = nppiWarpAffineQuad_8u_C3R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi, tmp_src_quad,
                                                dst->data[0], dst->linesize[0], tmp_dst_roi, tmp_dst_quad,
                                                (int)inter_mode);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB32 || src->format == (int)AV_PIX_FMT_BGR32)
        {

            npp_chk = nppiWarpAffineQuad_8u_C4R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi, tmp_src_quad,
                                                dst->data[0], dst->linesize[0], tmp_dst_roi, tmp_dst_quad,
                                                (int)inter_mode);
        }
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

#endif //__cuda_cuda_h__