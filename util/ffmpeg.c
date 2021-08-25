/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/ffmpeg.h"

#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H) && defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H)

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

/*------------------------------------------------------------------------------------------------*/

AVCodec *abcdk_avcodec_find(const char *name,int encode)
{
    AVCodec *ctx = NULL;

    assert(name != NULL);
    assert(*name != '\0');

    avcodec_register_all();

    ctx = (encode ? avcodec_find_encoder_by_name(name) : avcodec_find_decoder_by_name(name));

    return ctx;
}

AVCodec *abcdk_avcodec_find2(enum AVCodecID id,int encode)
{
    AVCodec *ctx = NULL;

    assert(id > AV_CODEC_ID_NONE);

    avcodec_register_all();

    if (id == AV_CODEC_ID_HEVC)
        ctx = abcdk_avcodec_find(encode?"hevc_nvenc":"hevc_cuvid", encode);
    if (id == AV_CODEC_ID_H264)
        ctx = abcdk_avcodec_find(encode?"h264_nvenc":"h264_cuvid", encode);
    
    if (!ctx)
        ctx = (encode ? avcodec_find_encoder(id) : avcodec_find_decoder(id));
    
    return ctx;
}

void abcdk_avcodec_show_options(AVCodec *ctx)
{
    assert(ctx != NULL);

    if (ctx->priv_class)
        av_opt_show2((void *)&ctx->priv_class, NULL, -1, 0);
    else
        av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->long_name ? ctx->long_name : ctx->name));
}

void abcdk_avcodec_free(AVCodecContext **ctx)
{
    assert(ctx != NULL);

    if (*ctx)
        avcodec_close(*ctx);

    avcodec_free_context(ctx);
}

AVCodecContext *abcdk_avcodec_alloc(const AVCodec *ctx)
{
    assert(ctx != NULL);

    return avcodec_alloc_context3(ctx);
}

int abcdk_avcodec_open(AVCodecContext *ctx, AVDictionary **dict)
{
    int chk = -1;

    assert(ctx != NULL);
    assert(ctx->codec != NULL);

    /*如果是编码器，填写默认值。*/
    if (av_codec_is_encoder(ctx->codec))
    {
        if (dict)
        {
            if (ctx->codec_id == AV_CODEC_ID_H265)
                av_dict_set(dict, "x265-params", "bframes=0", 0);
            else if (ctx->codec_id == AV_CODEC_ID_H264)
                av_dict_set(dict, "x264opts", "bframes=0", 0);
        }
    }

    chk = avcodec_open2(ctx, NULL, dict);

    return chk;
}

int abcdk_avcodec_decode(AVCodecContext *ctx, AVFrame *out,const AVPacket *in)
{
    int got = -1;

    assert(ctx != NULL && out != NULL && in != NULL);

    /*No output.*/
    got = 0;
    
    if (ctx->codec->type == AVMEDIA_TYPE_VIDEO)
    {
        if (avcodec_decode_video2(ctx, out, &got, in) < 0)
            got = -1;
    }
    else if (ctx->codec->type == AVMEDIA_TYPE_AUDIO)
    {
        if (avcodec_decode_audio4(ctx, out, &got, in) < 0)
            got = -1;
    }
    else
    {
        got = -2;
    }

    return got;
}

int abcdk_avcodec_encode(AVCodecContext *ctx, AVPacket *out, const AVFrame *in)
{
    int got = -1;

    assert(ctx != NULL && out != NULL && in != NULL);

    /*No output.*/
    got = 0;

    if (ctx->codec->type == AVMEDIA_TYPE_VIDEO)
    {
        if (avcodec_encode_video2(ctx, out, in, &got) != 0)
            got = -1;
    }
    else if (ctx->codec->type == AVMEDIA_TYPE_AUDIO)
    {
        if (avcodec_encode_audio2(ctx, out, in, &got) != 0)
            got = -1;
    }
    else
    {
        got = -2;
    }

    return got;
}

void abcdk_avcodec_video_encode_prepare(AVCodecContext *ctx,int fps,int width,int height,int gop_size,int oformat_flags)
{
    assert(ctx != NULL && fps > 0 && width > 0 && height > 0);
    assert(ctx->codec != NULL);
    assert(ctx->codec->pix_fmts[0] != AV_PIX_FMT_NONE);

    /*-------------Copy from OpenCV----begin------------------*/

    int frame_rate = (int)(fps + 0.5);
    int frame_rate_base = 1;
    while (fabs(((double)frame_rate / frame_rate_base) - fps) > 0.001)
    {
        frame_rate_base *= 10;
        frame_rate = (int)(fps * frame_rate_base + 0.5);
    }

    ctx->time_base.den = frame_rate;
    ctx->time_base.num = frame_rate_base;

    /* adjust time base for supported framerates */
    if (ctx->codec && ctx->codec->supported_framerates)
    {
        const AVRational *p = ctx->codec->supported_framerates;
        AVRational req = {frame_rate, frame_rate_base};
        const AVRational *best = NULL;
        AVRational best_error = {INT_MAX, 1};
        for (; p->den != 0; p++)
        {
            AVRational error = av_sub_q(req, *p);
            if (error.num < 0)
                error.num *= -1;
            if (av_cmp_q(error, best_error) < 0)
            {
                best_error = error;
                best = p;
            }
        }

        if (best)
        {
            ctx->time_base.den = best->num;
            ctx->time_base.num = best->den;
        }
    }
    /*-------------Copy from OpenCV-----end---------------*/

    ctx->framerate.den = ctx->time_base.num;
    ctx->framerate.num = ctx->time_base.den;
    ctx->width = width;
    ctx->height = height;
    ctx->gop_size = (gop_size > 0 ? gop_size : ctx->time_base.den);

    if (ctx->codec_id == AV_CODEC_ID_H265 || ctx->codec_id == AV_CODEC_ID_H264 || ctx->codec_id == AV_CODEC_ID_MJPEG)
        ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    else
        ctx->pix_fmt = ctx->codec->pix_fmts[0];
   
    if (oformat_flags & AVFMT_GLOBALHEADER)
        ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
}

/*------------------------------------------------------------------------------------------------*/


#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H && AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H
