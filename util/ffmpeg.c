/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/ffmpeg.h"


#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H)

/*------------------------------------------------------------------------------------------------*/


static struct _abcdk_av_log_dict
{
    int av_log_level;
    int sys_log_level;
} abcdk_av_log_dict[] = {
    {AV_LOG_PANIC, LOG_ERR},
    {AV_LOG_FATAL, LOG_ERR},
    {AV_LOG_ERROR, LOG_ERR},
    {AV_LOG_WARNING, LOG_WARNING},
    {AV_LOG_INFO, LOG_INFO},
    {AV_LOG_VERBOSE, LOG_DEBUG},
    {AV_LOG_DEBUG, LOG_DEBUG},
    {AV_LOG_TRACE, LOG_DEBUG}
};

static void _abcdk_av_log_cb(void *opaque, int level, const char *fmt, va_list v)
{
    int sys_level = LOG_DEBUG;

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_av_log_dict); i++)
    {
        if (abcdk_av_log_dict[i].av_log_level != level)
            continue;

        sys_level = abcdk_av_log_dict[i].sys_log_level;
    }
    
    vsyslog(sys_level,fmt,v);
}

void abcdk_av_log2syslog()
{
    av_log_set_callback(_abcdk_av_log_cb);
}

/*------------------------------------------------------------------------------------------------*/

int abcdk_av_image_pixfmt_bits(enum AVPixelFormat pixfmt, int padded)
{
    const AVPixFmtDescriptor *desc;

    assert(ABCDK_AVPIXFMT_CHECK(pixfmt));

    desc = av_pix_fmt_desc_get(pixfmt);
    if (desc)
        return (padded ? av_get_padded_bits_per_pixel(desc) : av_get_bits_per_pixel(desc));

    return -1;
}

const char *abcdk_av_image_pixfmt_name(enum AVPixelFormat pixfmt)
{
    const AVPixFmtDescriptor *desc;

    if (ABCDK_AVPIXFMT_CHECK(pixfmt))
    {
        desc = av_pix_fmt_desc_get(pixfmt);
        if (desc)
            return av_get_pix_fmt_name(pixfmt);
    }

    return "Unknown";
}

int abcdk_av_image_fill_heights(int heights[4], int height, enum AVPixelFormat pixfmt)
{
    const AVPixFmtDescriptor *desc;
    int h;
    int planes_nb;

    assert(heights != NULL && height > 0 && ABCDK_AVPIXFMT_CHECK(pixfmt));

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

int abcdk_av_image_fill_strides(int strides[4],int width,int height,enum AVPixelFormat pixfmt,int align)
{
    int stride_nb;

    assert(strides != NULL && width > 0 && height > 0 && ABCDK_AVPIXFMT_CHECK(pixfmt));

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

int abcdk_av_image_fill_strides2(abcdk_av_image_t *img,int align)
{
    assert (img != NULL);

    return abcdk_av_image_fill_strides(img->strides, img->width, img->height, img->pixfmt, align);
}

int abcdk_av_image_fill_pointers(uint8_t *datas[4], const int strides[4], int height, enum AVPixelFormat pixfmt, void *buffer)
{
    int size;

    assert(datas != NULL && strides != NULL && height > 0 && ABCDK_AVPIXFMT_CHECK(pixfmt));

    size = av_image_fill_pointers(datas, pixfmt, height, (uint8_t *)buffer, strides);

    /*只是计算大小，清空无效指针。*/
    if (!buffer)
        memset(datas, 0, sizeof(uint8_t *));

    return size;
}

int abcdk_av_image_fill_pointers2(abcdk_av_image_t *img,void *buffer)
{
    assert (img != NULL);

    return abcdk_av_image_fill_pointers(img->datas, img->strides, img->height, img->pixfmt, buffer);
}

int abcdk_av_image_size(const int strides[4], int height, enum AVPixelFormat pixfmt)
{
    uint8_t *datas[4] = {0};

    return abcdk_av_image_fill_pointers(datas, strides, height, pixfmt, NULL);
}

int abcdk_av_image_size2(int width,int height,enum AVPixelFormat pixfmt,int align)
{
    int strides[4] = {0};
    int chk;

    chk = abcdk_av_image_fill_strides(strides,width,height,pixfmt,align);
    if(chk<=0)
        return chk;

    return abcdk_av_image_size(strides,height,pixfmt);
}

int abcdk_av_image_size3(const abcdk_av_image_t *img)
{
    assert (img != NULL);

    return abcdk_av_image_size(img->strides, img->height, img->pixfmt);
}

void abcdk_av_image_copy(uint8_t *dst_datas[4], int dst_strides[4], const uint8_t *src_datas[4], const int src_strides[4],
                         int width, int height, enum AVPixelFormat pixfmt)
{
    assert(dst_datas != NULL && dst_strides != NULL);
    assert(src_datas != NULL && src_strides != NULL);
    assert(width > 0 && height > 0 && ABCDK_AVPIXFMT_CHECK(pixfmt));

    av_image_copy(dst_datas, dst_strides, src_datas, src_strides, pixfmt, width, height);
}

void abcdk_av_image_copy2(abcdk_av_image_t *dst, const abcdk_av_image_t *src)
{
    assert(dst != NULL && src != NULL);

    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->pixfmt == src->pixfmt);

    abcdk_av_image_copy(dst->datas,dst->strides,(const uint8_t **)src->datas,src->strides,
                        src->width,src->height,src->pixfmt);
}

/*------------------------------------------------------------------------------------------------*/

void abcdk_sws_free(struct SwsContext **ctx)
{
    if(!ctx)
        return;

    if(*ctx)
        sws_freeContext(*ctx);

    /*Set to NULL(0).*/
    *ctx = NULL;
}

struct SwsContext *abcdk_sws_alloc(int src_width, int src_height, enum AVPixelFormat src_pixfmt,
                                   int dst_width, int dst_height, enum AVPixelFormat dst_pixfmt,
                                   int flags)
{
    assert(src_width > 0 && src_height > 0 && ABCDK_AVPIXFMT_CHECK(src_pixfmt));
    assert(dst_width > 0 && dst_height > 0 && ABCDK_AVPIXFMT_CHECK(dst_pixfmt));

    return sws_getContext(src_width, src_height, src_pixfmt,
                          dst_width, dst_height, dst_pixfmt,
                          flags, NULL, NULL, NULL);
}

struct SwsContext *abcdk_sws_alloc2(const abcdk_av_image_t *src, const abcdk_av_image_t *dst, int flags)
{
    assert(dst != NULL && src != NULL);

    return abcdk_sws_alloc(src->width, src->height, src->pixfmt,
                           dst->width, dst->height, dst->pixfmt,
                           flags);
}

#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H

/*------------------------------------------------------------------------------------------------*/