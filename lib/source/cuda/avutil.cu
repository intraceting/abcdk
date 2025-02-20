/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"


#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

int abcdk_cuda_avimage_copy(uint8_t *dst_datas[4], int dst_strides[4], int dst_in_host,
                            const uint8_t *src_datas[4], const int src_strides[4], int src_in_host,
                            int width, int height, enum AVPixelFormat pixfmt)
{
    int real_stride[4] = {0};
    int real_height[4] = {0};
    int chk;

    assert(dst_datas != NULL && dst_strides != NULL);
    assert(src_datas != NULL && src_strides != NULL);
    assert(width > 0 && height > 0);
    assert(AV_PIX_FMT_NONE < pixfmt && pixfmt < AV_PIX_FMT_NB);

    abcdk_avimage_fill_strides(real_stride, width, height, pixfmt, 1);
    abcdk_avimage_fill_heights(real_height, height, pixfmt);

    for (int i = 0; i < 4; i++)
    {
        if (!src_datas[i])
            break;

        chk = abcdk_cuda_memcpy_2d(dst_datas[i], dst_strides[i], 0, 0, dst_in_host,
                                   src_datas[i], src_strides[i], 0, 0, src_in_host,
                                   real_stride[i], real_height[i]);
        if (chk != 0)
            return -1;
    }

    return 0;
}

CUmemorytype abcdk_cuda_avframe_memory_type(const AVFrame *src)
{
    assert(src != NULL);

    if (src->hw_frames_ctx && ABCDK_PTR2I64(src->hw_frames_ctx->data,0) == 0x123456789)
        return CU_MEMORYTYPE_DEVICE;

    return CU_MEMORYTYPE_UNIFIED;
}

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

    buf_ptr = abcdk_cuda_alloc_z(buf_size);
    if (!buf_ptr)
        return NULL;

    av_buffer = av_buffer_create((uint8_t *)buf_ptr, buf_size, _abcdk_cuda_avbuffer_free, NULL, 0);
    if (!av_buffer)
    {
        abcdk_cuda_free(&buf_ptr);
        return NULL;
    }

    av_frame = av_frame_alloc();
    if (!av_frame)
    {
        av_buffer_unref(&av_buffer);
        return NULL;
    }

    av_frame->buf[0] = av_buffer; // bind to array.

    av_frame->hw_frames_ctx = av_buffer_allocz(sizeof(int64_t));
    if(!av_frame->hw_frames_ctx)
    {
        av_frame_free(&av_frame);
        return NULL;
    }

    /*标志已经占用。*/
    ABCDK_PTR2I64(av_frame->hw_frames_ctx->data,0) = 0x123456789;

    chk_size = abcdk_avimage_fill_pointers(av_frame->data, strides, height, pixfmt, av_buffer->data);
    assert(buf_size == chk_size);

    av_frame->width = width;
    av_frame->height = height;
    av_frame->format = (int)pixfmt;

    /*copy strides to linesize.*/
    for (int i = 0; i < 4; i++)
        av_frame->linesize[i] = strides[i];

    return av_frame;
}

int abcdk_cuda_avframe_copy(AVFrame *dst, const AVFrame *src)
{
    int dst_in_host, src_in_host;
    int chk;

    assert(dst != NULL && src != NULL);

    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->format == src->format);

    dst_in_host = (abcdk_cuda_avframe_memory_type(dst) != CU_MEMORYTYPE_DEVICE);
    src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);

    chk = abcdk_cuda_avimage_copy(dst->data, dst->linesize, dst_in_host,
                                  (const uint8_t **)src->data, src->linesize, src_in_host,
                                  src->width, src->height, (enum AVPixelFormat)src->format);
    if (chk != 0)
        return -1;

    return 0;
}

AVFrame *abcdk_cuda_avframe_clone(int dst_in_host, const AVFrame *src)
{
    AVFrame *dst;
    int chk;

    assert(src != NULL);

    if (dst_in_host)
        dst = abcdk_avframe_alloc(src->width, src->height, (enum AVPixelFormat)src->format, 1);
    else
        dst = abcdk_cuda_avframe_alloc(src->width, src->height, (enum AVPixelFormat)src->format, 1);
    if (!dst)
        return NULL;

    chk = abcdk_cuda_avframe_copy(dst, src);
    if (chk != 0)
    {
        av_frame_free(&dst);
        return NULL;
    }

    return dst;
}

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__