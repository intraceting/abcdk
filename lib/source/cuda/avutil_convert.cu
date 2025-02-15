/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__
#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H)

static void _abcdk_cuda_avframe_cvt_free_tmp(AVFrame *dst, const AVFrame *src, AVFrame *tmp_dst, AVFrame *tmp_src)
{
    if (tmp_dst != dst)
        av_frame_free(&tmp_dst);

    if (tmp_src != src)
        av_frame_free(&tmp_src);
}

static int _abcdk_cuda_avframe_cvt_use_ffmpeg(AVFrame *dst, const AVFrame *src, int dst_in_host, int src_in_host)
{
    struct SwsContext *ctx = NULL;
    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    int chk = -1;

    ctx = abcdk_sws_alloc2(src, dst, 0);
    if (!ctx)
        return -1;

    tmp_dst = (!dst_in_host ? abcdk_cuda_avframe_clone(dst, 1, 0) : dst);
    tmp_src = (!src_in_host ? abcdk_cuda_avframe_clone(src, 1, 0) : (AVFrame *)src);

    if (!tmp_dst || !tmp_src)
    {
        _abcdk_cuda_avframe_cvt_free_tmp(dst, src, tmp_dst, tmp_src);
        abcdk_sws_free(&ctx);
        return -1;
    }

    chk = abcdk_sws_scale(ctx, src, dst);

    /*按需复制。*/
    if (chk > 0 && dst != tmp_dst)
        abcdk_cuda_avframe_copy(dst, tmp_dst, 0, 1);

    _abcdk_cuda_avframe_cvt_free_tmp(dst, src, tmp_dst, tmp_src);
    abcdk_sws_free(&ctx);

    if (chk <= 0)
        return -1;

    return 0;
}

static int _abcdk_cuda_avframe_cvt_use_nppi(AVFrame *dst, const AVFrame *src, int dst_in_host, int src_in_host)
{
    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;

    tmp_dst = (dst_in_host ? abcdk_cuda_avframe_clone(dst, 0, 1) : dst);
    tmp_src = (src_in_host ? abcdk_cuda_avframe_clone(src, 0, 1) : (AVFrame *)src);

    if (!tmp_dst || !tmp_src)
    {
        _abcdk_cuda_avframe_cvt_free_tmp(dst, src, tmp_dst, tmp_src);
        return -1;
    }

    NppiSize src_roi = {tmp_src->width, tmp_src->height};

    if (dst->format == (int)AV_PIX_FMT_GRAY8)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24)
        {
            npp_chk = nppiRGBToGray_8u_C3C1R(tmp_src->data[0], tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi);
        }
    }
    else if (dst->format == (int)AV_PIX_FMT_RGB24)
    {
        if (src->format == (int)AV_PIX_FMT_BGR24)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C3R(tmp_src->data[0], tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB32)
        {
            int dst_order[3] = {0, 1, 2};
            npp_chk = nppiSwapChannels_8u_C4C3R(tmp_src->data[0], tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_BGR32)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C4C3R(tmp_src->data[0], tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_YUVJ420P ||
                 src->format == (int)AV_PIX_FMT_YUV420P)
        {
            npp_chk = nppiYUV420ToRGB_8u_P3C3R(tmp_src->data, tmp_src->linesize, tmp_dst->data[0], tmp_dst->linesize[0], src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_NV12)
        {
            npp_chk = nppiNV12ToRGB_8u_P2C3R(tmp_src->data, tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_YUVJ422P ||
                 src->format == (int)AV_PIX_FMT_YUV422P)
        {
            npp_chk = nppiYUV422ToRGB_8u_P3C3R(tmp_src->data, tmp_src->linesize, tmp_dst->data[0], tmp_dst->linesize[0], src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_YUV444P ||
                 src->format == (int)AV_PIX_FMT_YUVJ444P)
        {
            npp_chk = nppiYCbCr444ToRGB_JPEG_8u_P3C3R(tmp_src->data, tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi);
        }
    }
    else if (dst->format == (int)AV_PIX_FMT_BGR24)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C3R(tmp_src->data[0], tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_BGR32)
        {
            int dst_order[3] = {0, 1, 2};
            npp_chk = nppiSwapChannels_8u_C4C3R(tmp_src->data[0], tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB32)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C4C3R(tmp_src->data[0], tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_YUVJ420P ||
                 src->format == (int)AV_PIX_FMT_YUV420P)
        {
            npp_chk = nppiYUV420ToBGR_8u_P3C3R(tmp_src->data, tmp_src->linesize, tmp_dst->data[0], tmp_dst->linesize[0], src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_NV12)
        {
            npp_chk = nppiNV12ToBGR_8u_P2C3R(tmp_src->data, tmp_src->linesize[0], tmp_dst->data[0], tmp_dst->linesize[0], src_roi);
        }
    }
    else if (dst->format == (int)AV_PIX_FMT_YUVJ420P ||
             dst->format == (int)AV_PIX_FMT_YUV420P)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24)
        {
            npp_chk = nppiRGBToYUV420_8u_C3P3R(tmp_src->data[0], tmp_src->linesize[0], tmp_dst->data, tmp_dst->linesize, src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_NV12)
        {
            npp_chk = nppiNV12ToYUV420_8u_P2P3R(tmp_src->data, tmp_src->linesize[0], tmp_dst->data, tmp_dst->linesize, src_roi);
        }
    }

    /*按需复制。*/
    if (npp_chk == NPP_SUCCESS && dst != tmp_dst)
        abcdk_cuda_avframe_copy(dst, tmp_dst, 1, 0);

    _abcdk_cuda_avframe_cvt_free_tmp(dst, src, tmp_dst, tmp_src);

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

int abcdk_cuda_avframe_convert(AVFrame *dst, const AVFrame *src, int dst_in_host, int src_in_host)
{
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->width == src->width && dst->height == src->height);

    if (dst->format == src->format)
    {
        chk = abcdk_cuda_avframe_copy(dst, src, dst_in_host, src_in_host);
        if (chk != 0)
            return -1;
    }
    else if (dst->format == (int)AV_PIX_FMT_GRAY8)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24)
            chk = _abcdk_cuda_avframe_cvt_use_nppi(dst, src, dst_in_host, src_in_host);
        else
            chk = _abcdk_cuda_avframe_cvt_use_ffmpeg(dst, src, dst_in_host, src_in_host);

        if (chk != 0)
            return -1;
    }
    else if (dst->format == (int)AV_PIX_FMT_RGB24)
    {
        if (src->format == (int)AV_PIX_FMT_BGR24 ||
            src->format == (int)AV_PIX_FMT_RGB32 ||
            src->format == (int)AV_PIX_FMT_BGR32 ||
            src->format == (int)AV_PIX_FMT_YUVJ420P ||
            src->format == (int)AV_PIX_FMT_YUV420P ||
            src->format == (int)AV_PIX_FMT_NV12 ||
            src->format == (int)AV_PIX_FMT_YUVJ422P ||
            src->format == (int)AV_PIX_FMT_YUV422P ||
            src->format == (int)AV_PIX_FMT_YUV444P ||
            src->format == (int)AV_PIX_FMT_YUVJ444P)
            chk = _abcdk_cuda_avframe_cvt_use_nppi(dst, src, dst_in_host, src_in_host);
        else
            chk = _abcdk_cuda_avframe_cvt_use_ffmpeg(dst, src, dst_in_host, src_in_host);

        if (chk != 0)
            return -1;
    }
    else if (dst->format == (int)AV_PIX_FMT_BGR24)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24 ||
            src->format == (int)AV_PIX_FMT_BGR32 ||
            src->format == (int)AV_PIX_FMT_RGB32 ||
            src->format == (int)AV_PIX_FMT_YUVJ420P ||
            src->format == (int)AV_PIX_FMT_YUV420P ||
            src->format == (int)AV_PIX_FMT_NV12)
            chk = _abcdk_cuda_avframe_cvt_use_nppi(dst, src, dst_in_host, src_in_host);
        else
            chk = _abcdk_cuda_avframe_cvt_use_ffmpeg(dst, src, dst_in_host, src_in_host);

        if (chk != 0)
            return -1;
    }
    else if (dst->format == (int)AV_PIX_FMT_YUVJ420P ||
             dst->format == (int)AV_PIX_FMT_YUV420P)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24 ||
            src->format == (int)AV_PIX_FMT_NV12)
            chk = _abcdk_cuda_avframe_cvt_use_nppi(dst, src, dst_in_host, src_in_host);
        else
            chk = _abcdk_cuda_avframe_cvt_use_ffmpeg(dst, src, dst_in_host, src_in_host);

        if (chk != 0)
            return -1;
    }
    else
    {
        chk = _abcdk_cuda_avframe_cvt_use_ffmpeg(dst, src, dst_in_host, src_in_host);

        if (chk != 0)
            return -1;
    }

    return 0;
}

#endif // defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H)
#endif //__cuda_cuda_h__