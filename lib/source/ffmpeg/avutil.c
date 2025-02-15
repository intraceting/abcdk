/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/ffmpeg/avutil.h"

#ifdef AVUTIL_AVUTIL_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

double abcdk_avmatch_r2d(AVRational r, double scale)
{
    double a = (r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den);

    return scale * a;
}

int abcdk_avimage_pixfmt_bits(enum AVPixelFormat pixfmt, int padded)
{
    const AVPixFmtDescriptor *desc;

    assert(pixfmt > AV_PIX_FMT_NONE);

    desc = av_pix_fmt_desc_get(pixfmt);
    if (desc)
        return (padded ? av_get_padded_bits_per_pixel(desc) : av_get_bits_per_pixel(desc));

    return -1;
}

const char *abcdk_avimage_pixfmt_name(enum AVPixelFormat pixfmt)
{
    const AVPixFmtDescriptor *desc;

    if (pixfmt > AV_PIX_FMT_NONE)
    {
        desc = av_pix_fmt_desc_get(pixfmt);
        if (desc)
            return av_get_pix_fmt_name(pixfmt);
    }

    return NULL;
}

int abcdk_avimage_fill_heights(int heights[4], int height, enum AVPixelFormat pixfmt)
{
    const AVPixFmtDescriptor *desc;
    int h;
    int planes_nb;

    assert(heights != NULL && height > 0 && pixfmt > AV_PIX_FMT_NONE);

    desc = av_pix_fmt_desc_get(pixfmt);
    if (!desc)
        return -1;

    planes_nb = 0;
    for (int i = 0; i < desc->nb_components; i++)
        planes_nb = FFMAX(planes_nb, desc->comp[i].plane + 1);

    if (planes_nb <= 4)
    {
        for (int i = 0; i < planes_nb; i++)
        {
            h = height;
            if (i == 1 || i == 2)
            {
                h = FF_CEIL_RSHIFT(height, desc->log2_chroma_h);
            }

            heights[i] = h;
        }
    }

    return planes_nb;
}

int abcdk_avimage_fill_strides(int strides[4],int width,int height,enum AVPixelFormat pixfmt,int align)
{
    int stride_nb;

    assert(strides != NULL && width > 0 && height > 0 && pixfmt > AV_PIX_FMT_NONE);

    if (av_image_fill_linesizes(strides, pixfmt, width) < 0)
        ABCDK_ERRNO_AND_RETURN1(EINVAL, -1);

    stride_nb = 0;
    for (int i = 0; i < 4; i++)
    {
        if (strides[i] <= 0)
            continue;

        strides[i] = abcdk_align(strides[i], align);
        stride_nb += 1;
    }

    return stride_nb;
}

int abcdk_avimage_fill_pointers(uint8_t *datas[4], const int strides[4], int height,
                                 enum AVPixelFormat pixfmt, void *buffer)
{
    int size;

    assert(datas != NULL && strides != NULL && height > 0 && pixfmt > AV_PIX_FMT_NONE);

    size = av_image_fill_pointers(datas, pixfmt, height, (uint8_t *)buffer, strides);

    /*只是计算大小，清空无效指针。*/
    if (!buffer)
        datas[0] = datas[1] = datas[2] = datas[3] = NULL;

    return size;
}

int abcdk_avimage_size(const int strides[4], int height, enum AVPixelFormat pixfmt)
{
    uint8_t *datas[4] = {0};

    return abcdk_avimage_fill_pointers(datas, strides, height, pixfmt, NULL);
}

int abcdk_avimage_size2(int width,int height,enum AVPixelFormat pixfmt,int align)
{
    int strides[4] = {0};
    int chk;

    chk = abcdk_avimage_fill_strides(strides,width,height,pixfmt,align);
    if(chk<=0)
        return chk;

    return abcdk_avimage_size(strides,height,pixfmt);
}

void abcdk_avimage_copy(uint8_t *dst_datas[4], int dst_strides[4], const uint8_t *src_datas[4],
                         const int src_strides[4], int width, int height, enum AVPixelFormat pixfmt)
{
    assert(dst_datas != NULL && dst_strides != NULL);
    assert(src_datas != NULL && src_strides != NULL);
    assert(width > 0 && height > 0 && pixfmt > AV_PIX_FMT_NONE);

    av_image_copy(dst_datas, dst_strides, src_datas, src_strides, pixfmt, width, height);
}


void abcdk_avframe_copy(AVFrame *dst, const AVFrame *src)
{
    assert(dst != NULL && src != NULL);

    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->format == src->format);

    abcdk_avimage_copy(dst->data,dst->linesize,(const uint8_t **)src->data,src->linesize,
                        src->width,src->height,src->format);
}

static void _abcdk_avbuffer_free(void *opaque, uint8_t *data) 
{
    if(data)
        av_free(data);
}

AVFrame *abcdk_avframe_alloc(int width,int height,enum AVPixelFormat pixfmt,int align)
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

    buf_ptr = av_malloc(buf_size);
    if (!buf_ptr)
        return NULL;

    av_buffer = av_buffer_create((uint8_t*)buf_ptr, buf_size, _abcdk_avbuffer_free, NULL, 0);
    if(!av_buffer)
    {
        av_free(buf_ptr);
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

static void _abcdk_avlog_callback(void* opaque, int level, const char* fmt, va_list v)
{
    int sys_level;
    if((AV_LOG_QUIET == level) || (AV_LOG_PANIC == level)  || (AV_LOG_FATAL == level) || (AV_LOG_ERROR == level))
        sys_level = LOG_ERR;
    else if(AV_LOG_WARNING == level)
        sys_level = LOG_WARNING;
    else if(AV_LOG_INFO == level)
        sys_level = LOG_INFO;
    else if(AV_LOG_VERBOSE == level)
        sys_level = LOG_DEBUG;
    else
        return ;
    
    abcdk_trace_vprintf(sys_level,fmt,v);
}

void abcdk_avlog_redirect2trace()
{
    av_log_set_callback(_abcdk_avlog_callback);
}

#pragma GCC diagnostic pop

#endif //AVUTIL_AVUTIL_H