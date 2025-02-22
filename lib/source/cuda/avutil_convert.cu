/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"


#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef SWSCALE_SWSCALE_H

static int _abcdk_cuda_avframe_convert_ffmpeg(AVFrame *dst, const AVFrame *src)
{
    struct SwsContext *ctx = NULL;
    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    int dst_in_host, src_in_host;
    int chk = -1;
    
    dst_in_host = (abcdk_cuda_avframe_memory_type(dst) != CU_MEMORYTYPE_DEVICE);
    src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);

    if(!src_in_host)
    {
        tmp_src = abcdk_cuda_avframe_clone(1, src);
        if(!tmp_src)
            return -1;

        chk = _abcdk_cuda_avframe_convert_ffmpeg(dst,tmp_src);
        av_frame_free(&tmp_src);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if(!dst_in_host)
    {
        tmp_dst = abcdk_cuda_avframe_clone(1, dst);
        if(!tmp_dst)
            return -1;

        chk = _abcdk_cuda_avframe_convert_ffmpeg(tmp_dst,src);
        if(chk == 0)
            abcdk_cuda_avframe_copy(dst,tmp_dst);
        av_frame_free(&tmp_dst);

        return chk;
    }

    ctx = abcdk_sws_alloc2(src, dst, 0);
    if (!ctx)
        return -1;

    chk = abcdk_sws_scale(ctx, src, dst);
    abcdk_sws_free(&ctx);

    if (chk <= 0)
        return -1;

    return 0;
}

static int _abcdk_cuda_avframe_convert(AVFrame *dst, const AVFrame *src)
{
    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    int dst_in_host, src_in_host;
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;
    int chk;

    dst_in_host = (abcdk_cuda_avframe_memory_type(dst) != CU_MEMORYTYPE_DEVICE);
    src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);

    if(src_in_host)
    {
        tmp_src = abcdk_cuda_avframe_clone(0, src);
        if(!tmp_src)
            return -1;

        chk = _abcdk_cuda_avframe_convert(dst,tmp_src);
        av_frame_free(&tmp_src);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if(dst_in_host)
    {
        tmp_dst = abcdk_cuda_avframe_clone(0, dst);
        if(!tmp_dst)
            return -1;

        chk = _abcdk_cuda_avframe_convert(tmp_dst,src);
        if(chk == 0)
            abcdk_cuda_avframe_copy(dst,tmp_dst);
        av_frame_free(&tmp_dst);

        return chk;
    }

    NppiSize src_roi = {src->width, src->height};

    if (dst->format == (int)AV_PIX_FMT_GRAY8)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24)
        {
            npp_chk = nppiRGBToGray_8u_C3C1R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi);
        }
        else
        {
            tmp_dst = abcdk_cuda_avframe_alloc(dst->width,dst->height,AV_PIX_FMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_avframe_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_avframe_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }
    else if (dst->format == (int)AV_PIX_FMT_RGB24)
    {
        if (src->format == (int)AV_PIX_FMT_BGR24)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB32)
        {
            int dst_order[3] = {0, 1, 2};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_BGR32)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_YUVJ420P ||
                 src->format == (int)AV_PIX_FMT_YUV420P)
        {
            npp_chk = nppiYUV420ToRGB_8u_P3C3R(src->data, (int *)src->linesize, dst->data[0], dst->linesize[0], src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_NV12)
        {
            npp_chk = nppiNV12ToRGB_8u_P2C3R(src->data, src->linesize[0], dst->data[0], dst->linesize[0], src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_YUVJ422P ||
                 src->format == (int)AV_PIX_FMT_YUV422P)
        {
            npp_chk = nppiYUV422ToRGB_8u_P3C3R(src->data, (int *)src->linesize, dst->data[0], dst->linesize[0], src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_YUV444P ||
                 src->format == (int)AV_PIX_FMT_YUVJ444P)
        {
            npp_chk = nppiYCbCr444ToRGB_JPEG_8u_P3C3R(src->data, src->linesize[0], dst->data[0], dst->linesize[0], src_roi);
        }
        else
        {
            return _abcdk_cuda_avframe_convert_ffmpeg(dst,src);
        }
    }
    else if (dst->format == (int)AV_PIX_FMT_BGR24)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_BGR32)
        {
            int dst_order[3] = {0, 1, 2};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB32)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
        }
        else if (src->format == (int)AV_PIX_FMT_YUVJ420P ||
                 src->format == (int)AV_PIX_FMT_YUV420P)
        {
            npp_chk = nppiYUV420ToBGR_8u_P3C3R(src->data, (int *)src->linesize, dst->data[0], dst->linesize[0], src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_NV12)
        {
            npp_chk = nppiNV12ToBGR_8u_P2C3R(src->data, src->linesize[0], dst->data[0], dst->linesize[0], src_roi);
        }
        else
        {
            tmp_dst = abcdk_cuda_avframe_alloc(dst->width,dst->height,AV_PIX_FMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_avframe_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_avframe_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }
    else if (dst->format == (int)AV_PIX_FMT_YUVJ420P ||
             dst->format == (int)AV_PIX_FMT_YUV420P)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24)
        {
            npp_chk = nppiRGBToYUV420_8u_C3P3R(src->data[0], src->linesize[0], dst->data, dst->linesize, src_roi);
        }
        else if (src->format == (int)AV_PIX_FMT_NV12)
        {
            npp_chk = nppiNV12ToYUV420_8u_P2P3R(src->data, src->linesize[0], dst->data, dst->linesize, src_roi);
        }
        else
        {
            tmp_dst = abcdk_cuda_avframe_alloc(dst->width,dst->height,AV_PIX_FMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_avframe_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_avframe_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }
    else if (dst->format == (int)AV_PIX_FMT_RGB32)
    {
        if (src->format == (int)AV_PIX_FMT_RGB24)
        {
            int dst_order[4] = {0, 1, 2, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order,0);
        }
        else if (src->format == (int)AV_PIX_FMT_BGR24)
        {
            int dst_order[4] = {2, 1, 0, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order,0);
        }
        else
        {
            tmp_dst = abcdk_cuda_avframe_alloc(dst->width,dst->height,AV_PIX_FMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_avframe_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_avframe_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }
    else if (dst->format == (int)AV_PIX_FMT_BGR32)
    {
        if (src->format == (int)AV_PIX_FMT_BGR24)
        {
            int dst_order[4] = {0, 1, 2, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order,0);
        }
        else if (src->format == (int)AV_PIX_FMT_RGB24)
        {
            int dst_order[4] = {2, 1, 0, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order,0);
        }
        else
        {
            tmp_dst = abcdk_cuda_avframe_alloc(dst->width,dst->height,AV_PIX_FMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_avframe_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_avframe_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

int abcdk_cuda_avframe_convert(AVFrame *dst, const AVFrame *src)
{
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->width == src->width && dst->height == src->height);

    if (dst->format == src->format)
    {
        chk = abcdk_cuda_avframe_copy(dst,src);
        if (chk != 0)
            return -1;
    }
    else
    {
        chk = _abcdk_cuda_avframe_convert(dst, src);
        if (chk != 0)
            return -1;
    }

    return 0;
}

#endif // SWSCALE_SWSCALE_H
#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__