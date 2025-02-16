/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

int abcdk_cuda_avframe_warpperspective(AVFrame *dst, const NppiRect *dst_roi,
                                        const AVFrame *src, const NppiRect *src_roi,
                                        const NppiPoint quad[4], const NppiRect *quad_roi,
                                        int back, NppiInterpolationMode inter_mode)
{

    NppiSize tmp_src_size = {0};
    NppiRect tmp_dst_roi = {0}, tmp_src_roi = {0}, tmp_quad_roi = {0};
    double tmp_quad[4][2], tmp_coeffs[3][3];
    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    int dst_in_host, src_in_host;
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->format == src->format);
    assert(src->format == (int)AV_PIX_FMT_GRAY8 ||
           src->format == (int)AV_PIX_FMT_RGB24 || src->format == (int)AV_PIX_FMT_BGR24 ||
           src->format == (int)AV_PIX_FMT_RGB32 || src->format == (int)AV_PIX_FMT_BGR32);
    assert(quad != NULL);

    dst_in_host = (abcdk_cuda_avframe_memory_type(dst) != CU_MEMORYTYPE_DEVICE);
    src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);

    if (src_in_host)
    {
        tmp_src = abcdk_cuda_avframe_clone(0, src);
        if (!tmp_src)
            return -1;

        chk = abcdk_cuda_avframe_warpperspective(dst, dst_roi, tmp_src, src_roi, quad, quad_roi, back, inter_mode);
        av_frame_free(&tmp_src);

        return chk;
    }

    if (dst_in_host)
    {
        tmp_dst = abcdk_cuda_avframe_clone(0, dst);
        if (!tmp_dst)
            return -1;

        chk = abcdk_cuda_avframe_warpperspective(tmp_dst, dst_roi, src, src_roi, quad, quad_roi, back, inter_mode);
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

    tmp_quad_roi.x = (quad_roi ? quad_roi->x : 0);
    tmp_quad_roi.y = (quad_roi ? quad_roi->y : 0);
    tmp_quad_roi.width = (quad_roi ? quad_roi->width : dst->width);
    tmp_quad_roi.height = (quad_roi ? quad_roi->height : dst->height);

    for (int i = 0; i < 4; i++)
    {
        tmp_quad[i][0] = quad[i].x;
        tmp_quad[i][1] = quad[i].y;
    }

    nppiGetPerspectiveTransform(tmp_quad_roi, tmp_quad, tmp_coeffs);

    if (src->format == (int)AV_PIX_FMT_GRAY8)
    {
        if (back)
        {
            /*多边型向矩形变换。*/
            npp_chk = nppiWarpPerspectiveBack_8u_C1R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                                     dst->data[0], dst->linesize[0], tmp_dst_roi,
                                                     tmp_coeffs, (int)inter_mode);
        }
        else
        {
            /*矩形向多边型变换。*/
            npp_chk = nppiWarpPerspective_8u_C1R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                                 dst->data[0], dst->linesize[0], tmp_dst_roi,
                                                 tmp_coeffs, (int)inter_mode);
        }
    }
    else if (src->format == (int)AV_PIX_FMT_RGB24 || src->format == (int)AV_PIX_FMT_BGR24)
    {
        if (back)
        {
            /*多边型向矩形变换。*/
            npp_chk = nppiWarpPerspectiveBack_8u_C3R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                                     dst->data[0], dst->linesize[0], tmp_dst_roi,
                                                     tmp_coeffs, (int)inter_mode);
        }
        else
        {
            /*矩形向多边型变换。*/
            npp_chk = nppiWarpPerspective_8u_C3R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                                 dst->data[0], dst->linesize[0], tmp_dst_roi,
                                                 tmp_coeffs, (int)inter_mode);
        }
    }
    else if (src->format == (int)AV_PIX_FMT_RGB32 || src->format == (int)AV_PIX_FMT_BGR32)
    {
        if (back)
        {
            /*多边型向矩形变换。*/
            npp_chk = nppiWarpPerspectiveBack_8u_C4R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                                     dst->data[0], dst->linesize[0], tmp_dst_roi,
                                                     tmp_coeffs, (int)inter_mode);
        }
        else
        {
            /*矩形向多边型变换。*/
            npp_chk = nppiWarpPerspective_8u_C4R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                                 dst->data[0], dst->linesize[0], tmp_dst_roi,
                                                 tmp_coeffs, (int)inter_mode);
        }
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

#endif //__cuda_cuda_h__