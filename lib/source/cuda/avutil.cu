/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

static void _abcdk_cuda_avbuffer_free(void *opaque, uint8_t *data) 
{
    abcdk_cuda_free((void **)&data);
}

AVFrame *abcdk_cuda_avframe_alloc(int width, int height, enum AVPixelFormat pixfmt, int align)
{
    AVBufferRef *av_buffer = NULL;
    AVFrame *av_frame = NULL;
    int strides[4] = {0};
    int buf_size;
    void *buf_ptr = NULL;
    int chk_size;

    assert(width > 0 && height > 0);
    assert(AV_PIX_FMT_NONE < pixfmt && pixfmt < AV_PIX_FMT_NB);
    
    if (abcdk_avimage_fill_strides(strides, width, height, pixfmt, align) <= 0)
        return NULL;

    buf_size = abcdk_avimage_size(strides, height, pixfmt);
    if (buf_size <= 0)
        return NULL;

    buf_ptr = abcdk_cuda_alloc(buf_size);
    if (!buf_ptr)
        return NULL;

    av_buffer = av_buffer_create((uint8_t*)buf_ptr, buf_size, _abcdk_cuda_avbuffer_free, NULL, 0);
    if(!av_buffer)
    {
        abcdk_cuda_free(&buf_ptr);
        return NULL;
    }

    av_frame = av_frame_alloc();
    if(!av_frame)
    {
        av_buffer_unref(&av_buffer);
        return NULL;
    }

    chk_size = abcdk_avimage_fill_pointers(av_frame->data, strides, height, pixfmt, av_buffer->data);
    assert(buf_size == chk_size);

    av_frame->width = width;
    av_frame->height = height;
    av_frame->format = (int)pixfmt;
    av_frame->buf[0] = av_buffer;//bind to array.

    /*copy strides to linesize.*/
    for (int i = 0; i < 4; i++)
        av_frame->linesize[i] = strides[i];

    return av_frame;
}

int abcdk_cuda_avimage_copy(uint8_t *dst_datas[4], int dst_strides[4],
                            const uint8_t *src_datas[4], const int src_strides[4],
                            int width, int height, enum AVPixelFormat pixfmt,
                            int dst_in_host, int src_in_host)
{
    int real_stride[4]= {0};
    int real_height[4]= {0};
    int chk;

    assert(dst_datas != NULL && dst_strides != NULL);
    assert(src_datas != NULL && src_strides != NULL);
    assert(width > 0 && height > 0);
    assert(AV_PIX_FMT_NONE < pixfmt && pixfmt < AV_PIX_FMT_NB);

    abcdk_avimage_fill_strides(real_stride,width,height,pixfmt,1);
    abcdk_avimage_fill_heights(real_height,height,pixfmt);

    for (int i = 0; i < 4; i++)
    {
        if (!src_datas[i])
            break;

        chk = abcdk_cuda_memcpy_2d(dst_datas[i], dst_strides[i], 0, 0,
                                   src_datas[i], src_strides[i], 0, 0,
                                   real_stride[i], real_height[i],
                                   dst_in_host, src_in_host);
        if (chk != 0)
            return -1;
    }

    return 0;
}

int abcdk_cuda_avframe_copy(AVFrame *dst, const AVFrame *src, int dst_in_host, int src_in_host)
{
    int chk;

    assert(dst != NULL && src != NULL);

    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->format == src->format);
   
    chk = abcdk_cuda_avimage_copy(dst->data,dst->linesize,(const uint8_t **)src->data,src->linesize,src->width,src->height,(enum AVPixelFormat)src->format,dst_in_host,src_in_host);
    if(chk != 0)
        return -1;

    return 0;
}

AVFrame *abcdk_cuda_avframe_clone(const AVFrame *src, int dst_in_host, int src_in_host)
{
    AVFrame *dst;
    int chk;

    assert(src != NULL);

    if(dst_in_host)
        dst = abcdk_avframe_alloc(src->width,src->height,(enum AVPixelFormat)src->format,4);
    else 
        dst = abcdk_cuda_avframe_alloc(src->width,src->height,(enum AVPixelFormat)src->format,4);
    if(!dst)
        return NULL;

    chk = abcdk_cuda_avframe_copy(dst,src,dst_in_host,src_in_host);
    if(chk != 0)
    {
        av_frame_free(&dst);
        return NULL;
    }

    return dst;
}

#endif //AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__