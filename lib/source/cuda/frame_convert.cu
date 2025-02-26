/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/frame.h"

#ifdef __cuda_cuda_h__

static int _abcdk_cuda_frame_convert_cpu(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    abcdk_media_frame_t *tmp_dst = NULL, *tmp_src = NULL;
    int dst_in_host, src_in_host;
    int chk = -1;
    
    dst_in_host = (dst->tag == ABCDK_MEDIA_TAG_HOST);
    src_in_host = (src->tag == ABCDK_MEDIA_TAG_HOST);

    if(!src_in_host)
    {
        tmp_src = abcdk_cuda_frame_clone(1, src);
        if(!tmp_src)
            return -1;

        chk = _abcdk_cuda_frame_convert_cpu(dst,tmp_src);

        abcdk_media_frame_free(&tmp_src);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if(!dst_in_host)
    {
        tmp_dst = abcdk_cuda_frame_clone(1, dst);
        if(!tmp_dst)
            return -1;

        chk = _abcdk_cuda_frame_convert_cpu(tmp_dst,src);
        if(chk == 0)
            abcdk_cuda_frame_copy(dst,tmp_dst);

        abcdk_media_frame_free(&tmp_dst);

        return chk;
    }

    chk = abcdk_media_frame_convert(dst,src);
    if (chk <= 0)
        return -1;

    return 0;
}

static int _abcdk_cuda_frame_convert(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    AVFrame *tmp_dst = NULL, *tmp_src = NULL;
    int dst_in_host, src_in_host;
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;
    int chk;

    dst_in_host = (dst->tag == ABCDK_MEDIA_TAG_HOST);
    src_in_host = (src->tag == ABCDK_MEDIA_TAG_HOST);

    if(src_in_host)
    {
        tmp_src = abcdk_cuda_frame_clone(0, src);
        if(!tmp_src)
            return -1;

        chk = _abcdk_cuda_frame_convert(dst,tmp_src);
        abcdk_media_frame_free(&tmp_src);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if(dst_in_host)
    {
        tmp_dst = abcdk_cuda_frame_clone(0, dst);
        if(!tmp_dst)
            return -1;

        chk = _abcdk_cuda_frame_convert(tmp_dst,src);
        if(chk == 0)
            abcdk_cuda_frame_copy(dst,tmp_dst);

        abcdk_media_frame_free(&tmp_dst);

        return chk;
    }

    NppiSize src_roi = {src->width, src->height};

    if (dst->format == ABCDK_MEDIA_PIXFMT_GRAY8)
    {
        if (src->format == ABCDK_MEDIA_PIXFMT_RGB24)
        {
            npp_chk = nppiRGBToGray_8u_C3C1R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi);
        }
        else
        {
            tmp_dst = abcdk_cuda_frame_create(dst->width,dst->height,AV_PIX_FMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_frame_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_frame_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }
    else if (dst->format == ABCDK_MEDIA_PIXFMT_RGB24)
    {
        if (src->format == ABCDK_MEDIA_PIXFMT_BGR24)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_RGB32)
        {
            int dst_order[3] = {0, 1, 2};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_BGR32)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_YUVJ420P ||
                 src->format == ABCDK_MEDIA_PIXFMT_YUV420P)
        {
            npp_chk = nppiYUV420ToRGB_8u_P3C3R(src->data, (int *)src->stride, dst->data[0], dst->stride[0], src_roi);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_NV12)
        {
            npp_chk = nppiNV12ToRGB_8u_P2C3R(src->data, src->stride[0], dst->data[0], dst->stride[0], src_roi);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_YUVJ422P ||
                 src->format == ABCDK_MEDIA_PIXFMT_YUV422P)
        {
            npp_chk = nppiYUV422ToRGB_8u_P3C3R(src->data, (int *)src->stride, dst->data[0], dst->stride[0], src_roi);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_YUV444P ||
                 src->format == ABCDK_MEDIA_PIXFMT_YUVJ444P)
        {
            npp_chk = nppiYCbCr444ToRGB_JPEG_8u_P3C3R(src->data, src->stride[0], dst->data[0], dst->stride[0], src_roi);
        }
        else
        {
            return _abcdk_cuda_frame_convert_cpu(dst,src);
        }
    }
    else if (dst->format == ABCDK_MEDIA_PIXFMT_BGR24)
    {
        if (src->format == ABCDK_MEDIA_PIXFMT_RGB24)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_BGR32)
        {
            int dst_order[3] = {0, 1, 2};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_RGB32)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_YUVJ420P ||
                 src->format == ABCDK_MEDIA_PIXFMT_YUV420P)
        {
            npp_chk = nppiYUV420ToBGR_8u_P3C3R(src->data, (int *)src->stride, dst->data[0], dst->stride[0], src_roi);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_NV12)
        {
            npp_chk = nppiNV12ToBGR_8u_P2C3R(src->data, src->stride[0], dst->data[0], dst->stride[0], src_roi);
        }
        else
        {
            tmp_dst = abcdk_cuda_frame_create(dst->width,dst->height,ABCDK_MEDIA_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_frame_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_frame_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }
    else if (dst->format == ABCDK_MEDIA_PIXFMT_YUVJ420P ||
             dst->format == ABCDK_MEDIA_PIXFMT_YUV420P)
    {
        if (src->format == ABCDK_MEDIA_PIXFMT_RGB24)
        {
            npp_chk = nppiRGBToYUV420_8u_C3P3R(src->data[0], src->stride[0], dst->data, dst->stride, src_roi);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_NV12)
        {
            npp_chk = nppiNV12ToYUV420_8u_P2P3R(src->data, src->stride[0], dst->data, dst->stride, src_roi);
        }
        else
        {
            tmp_dst = abcdk_cuda_frame_create(dst->width,dst->height,ABCDK_MEDIA_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_frame_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_frame_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }
    else if (dst->format == ABCDK_MEDIA_PIXFMT_RGB32)
    {
        if (src->format == ABCDK_MEDIA_PIXFMT_RGB24)
        {
            int dst_order[4] = {0, 1, 2, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order,0);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_BGR24)
        {
            int dst_order[4] = {2, 1, 0, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order,0);
        }
        else
        {
            tmp_dst = abcdk_cuda_frame_create(dst->width,dst->height,ABCDK_MEDIA_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_frame_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_frame_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }
    else if (dst->format == ABCDK_MEDIA_PIXFMT_BGR32)
    {
        if (src->format == ABCDK_MEDIA_PIXFMT_BGR24)
        {
            int dst_order[4] = {0, 1, 2, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order,0);
        }
        else if (src->format == ABCDK_MEDIA_PIXFMT_RGB24)
        {
            int dst_order[4] = {2, 1, 0, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order,0);
        }
        else
        {
            tmp_dst = abcdk_cuda_frame_create(dst->width,dst->height,ABCDK_MEDIA_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_cuda_frame_convert(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_cuda_frame_convert(dst, tmp_dst);

            av_frame_free(&tmp_dst);
            return chk;
        }
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

int abcdk_cuda_frame_convert(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_MEDIA_TAG_HOST || dst->tag == ABCDK_MEDIA_TAG_CUDA);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST || src->tag == ABCDK_MEDIA_TAG_CUDA);
    assert(dst->width == src->width);
    assert(dst->height == src->height);

    if (dst->format == src->format)
    {
        chk = abcdk_cuda_frame_copy(dst,src);
        if (chk != 0)
            return -1;
    }
    else
    {
        chk = _abcdk_cuda_frame_convert(dst, src);
        if (chk != 0)
            return -1;
    }

    return 0;
}

#else //__cuda_cuda_h__

int abcdk_cuda_frame_convert(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

#endif //__cuda_cuda_h__