/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/image.h"


__BEGIN_DECLS

#ifdef __cuda_cuda_h__

static int _abcdk_torch_image_convert_cuda_use_cpu(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src)
{
    abcdk_torch_image_t *tmp_dst = NULL, *tmp_src = NULL;
    int chk = -1;

    if(src->tag != ABCDK_TORCH_TAG_HOST)
    {
        tmp_src = abcdk_torch_image_clone_cuda(1, src);
        if(!tmp_src)
            return -1;

        chk = _abcdk_torch_image_convert_cuda_use_cpu(dst,tmp_src);

        abcdk_torch_image_free_host(&tmp_src);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if(dst->tag != ABCDK_TORCH_TAG_HOST)
    {
        tmp_dst = abcdk_torch_image_create_host(dst->width,dst->height,dst->pixfmt,1);
        if(!tmp_dst)
            return -1;

        chk = _abcdk_torch_image_convert_cuda_use_cpu(tmp_dst,src);
        if(chk == 0)
            abcdk_torch_image_copy_cuda(dst,tmp_dst);

        abcdk_torch_image_free_host(&tmp_dst);

        return chk;
    }

    chk = abcdk_torch_image_convert_host(dst,src);
    if (chk != 0)
        return -1;

    return 0;
}

static int _abcdk_torch_image_convert_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src)
{
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;

    NppiSize src_roi = {src->width, src->height};

    /* 颜色变换矩阵（标准 RGB → YUV 转换）*/
    Npp32f rgb_to_yuv_twist[3][4] = {
        {0.299f, 0.587f, 0.114f, 0.0f},     // Y
        {-0.169f, -0.331f, 0.500f, 128.0f}, // U
        {0.500f, -0.419f, -0.081f, 128.0f}  // V
    };

    int chk;

    if (dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
        {
            npp_chk = nppiRGBToGray_8u_C3C1R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi);
        }
        else
        {
            tmp_dst = abcdk_torch_image_create_cuda(dst->width,dst->height, ABCDK_TORCH_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_torch_image_convert_cuda(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_torch_image_convert_cuda(dst, tmp_dst);

            abcdk_torch_image_free_cuda(&tmp_dst);
            return chk;
        }
    }
    else if (dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_BGR24)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB32)
        {
            int dst_order[3] = {0, 1, 2};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_BGR32)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_YUV420P)
        {
            npp_chk = nppiYUV420ToRGB_8u_P3C3R(src->data, (int *)src->stride, dst->data[0], dst->stride[0], src_roi);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_YUV422P)
        {
            npp_chk = nppiYUV422ToRGB_8u_P3C3R(src->data, (int *)src->stride, dst->data[0], dst->stride[0], src_roi);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_YUV444P)
        {
            npp_chk = nppiYCbCr444ToRGB_JPEG_8u_P3C3R(src->data, src->stride[0], dst->data[0], dst->stride[0], src_roi);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_NV12)
        {
            npp_chk = nppiNV12ToRGB_8u_P2C3R(src->data, src->stride[0], dst->data[0], dst->stride[0], src_roi);
        }
        else
        {
            return _abcdk_torch_image_convert_cuda_use_cpu(dst,src);
        }
    }
    else if (dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_BGR32)
        {
            int dst_order[3] = {0, 1, 2};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB32)
        {
            int dst_order[3] = {2, 1, 0};
            npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_YUV420P)
        {
            npp_chk = nppiYUV420ToBGR_8u_P3C3R(src->data, (int *)src->stride, dst->data[0], dst->stride[0], src_roi);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_NV12)
        {
            npp_chk = nppiNV12ToBGR_8u_P2C3R(src->data, src->stride[0], dst->data[0], dst->stride[0], src_roi);
        }
        else
        {
            tmp_dst = abcdk_torch_image_create_cuda(dst->width,dst->height,ABCDK_TORCH_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_torch_image_convert_cuda(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_torch_image_convert_cuda(dst, tmp_dst);

            abcdk_torch_image_free_cuda(&tmp_dst);
            return chk;
        }
    }
    else if (dst->pixfmt == ABCDK_TORCH_PIXFMT_YUV420P)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
        {
            npp_chk = nppiRGBToYUV420_8u_C3P3R(src->data[0], src->stride[0], dst->data, dst->stride, src_roi);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_NV12)
        {
            npp_chk = nppiNV12ToYUV420_8u_P2P3R(src->data, src->stride[0], dst->data, dst->stride, src_roi);
        }
        else
        {
            tmp_dst = abcdk_torch_image_create_cuda(dst->width,dst->height,ABCDK_TORCH_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_torch_image_convert_cuda(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_torch_image_convert_cuda(dst, tmp_dst);

            abcdk_torch_image_free_cuda(&tmp_dst);
            return chk;
        }
    }
    else if (dst->pixfmt == ABCDK_TORCH_PIXFMT_YUV422P)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
        {
            npp_chk = nppiRGBToYUV422_8u_C3P3R(src->data[0], src->stride[0], dst->data, dst->stride, src_roi);
        }
        else
        {
            tmp_dst = abcdk_torch_image_create_cuda(dst->width,dst->height,ABCDK_TORCH_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_torch_image_convert_cuda(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_torch_image_convert_cuda(dst, tmp_dst);

            abcdk_torch_image_free_cuda(&tmp_dst);
            return chk;
        }
    }
    else if (dst->pixfmt == ABCDK_TORCH_PIXFMT_YUV444P)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
        {
            npp_chk = nppiRGBToYCbCr444_JPEG_8u_C3P3R(src->data[0], src->stride[0], dst->data, dst->stride[0], src_roi);
        }
        else
        {
            tmp_dst = abcdk_torch_image_create_cuda(dst->width,dst->height,ABCDK_TORCH_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_torch_image_convert_cuda(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_torch_image_convert_cuda(dst, tmp_dst);

            abcdk_torch_image_free_cuda(&tmp_dst);
            return chk;
        }
    }
    else if (dst->pixfmt == ABCDK_TORCH_PIXFMT_NV12)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
        {
            NppStreamContext stream_ctx = { 0 };
            npp_chk = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(src->data[0],src->stride[0],dst->data,dst->stride,src_roi,rgb_to_yuv_twist,stream_ctx);
        }
        else
        {
            tmp_dst = abcdk_torch_image_create(dst->width,dst->height,ABCDK_TORCH_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_torch_image_convert_cuda(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_torch_image_convert_cuda(dst, tmp_dst);

            abcdk_torch_image_free_cuda(&tmp_dst);
            return chk;
        }
    }
    else if (dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
        {
            int dst_order[4] = {0, 1, 2, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order,0);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_BGR24)
        {
            int dst_order[4] = {2, 1, 0, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order,0);
        }
        else
        {
            tmp_dst = abcdk_torch_image_create_cuda(dst->width,dst->height,ABCDK_TORCH_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_torch_image_convert_cuda(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_torch_image_convert_cuda(dst, tmp_dst);

            abcdk_torch_image_free_cuda(&tmp_dst);
            return chk;
        }
    }
    else if (dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32)
    {
        if (src->pixfmt == ABCDK_TORCH_PIXFMT_BGR24)
        {
            int dst_order[4] = {0, 1, 2, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order,0);
        }
        else if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
        {
            int dst_order[4] = {2, 1, 0, 3};
            npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->stride[0], dst->data[0], dst->stride[0], src_roi, dst_order,0);
        }
        else
        {
            tmp_dst = abcdk_torch_image_create_cuda(dst->width,dst->height,ABCDK_TORCH_PIXFMT_RGB24,1);
            if(!tmp_dst)
                return -1;

            chk = _abcdk_torch_image_convert_cuda(tmp_dst, src);
            if (chk == 0)
                chk = _abcdk_torch_image_convert_cuda(dst, tmp_dst);

            abcdk_torch_image_free_cuda(&tmp_dst);
            return chk;
        }
    }
    else
    {
        return _abcdk_torch_image_convert_cuda_use_cpu(dst,src);
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

int abcdk_torch_image_convert_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src)
{
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->tag == ABCDK_TORCH_TAG_CUDA);
    assert(dst->width == src->width);
    assert(dst->height == src->height);

    if (dst->pixfmt == src->pixfmt)
    {
        chk = abcdk_torch_image_copy_cuda(dst,src);
        if (chk != 0)
            return -1;
    }
    else
    {
        chk = _abcdk_torch_image_convert_cuda(dst, src);
        if (chk != 0)
            return -1;
    }

    return 0;
}

#else //__cuda_cuda_h__

int abcdk_torch_image_convert_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

#endif //__cuda_cuda_h__


__END_DECLS