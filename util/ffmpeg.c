/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/ffmpeg.h"

#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H) && defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

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

#define ABCDK_AVPIXFMT_CHECK(pixfmt)   ((pixfmt) > AV_PIX_FMT_NONE && (pixfmt) < AV_PIX_FMT_NB)

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

int abcdk_av_image_fill_strides2(abcdk_image_t *img,int align)
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

int abcdk_av_image_fill_pointers2(abcdk_image_t *img,void *buffer)
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

int abcdk_av_image_size3(const abcdk_image_t *img)
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

void abcdk_av_image_copy2(abcdk_image_t *dst, const abcdk_image_t *src)
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

struct SwsContext *abcdk_sws_alloc2(const abcdk_image_t *src, const abcdk_image_t *dst, int flags)
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

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    avcodec_register_all();
#endif

    ctx = (encode ? avcodec_find_encoder_by_name(name) : avcodec_find_decoder_by_name(name));

    return ctx;
}

AVCodec *abcdk_avcodec_find2(enum AVCodecID id,int encode)
{
    AVCodec *ctx = NULL;

    assert(id > AV_CODEC_ID_NONE);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100)
    avcodec_register_all();
#endif

    if (id == AV_CODEC_ID_HEVC)
        ctx = abcdk_avcodec_find(encode?"hevc_nvenc":"hevc_cuvid", encode);
    else if (id == AV_CODEC_ID_H264)
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

AVCodecContext *abcdk_avcodec_alloc2(const char *name,int encode)
{
    return abcdk_avcodec_alloc(abcdk_avcodec_find(name,encode));
}

AVCodecContext *abcdk_avcodec_alloc3(enum AVCodecID id,int encode)
{
    return abcdk_avcodec_alloc(abcdk_avcodec_find2(id,encode));
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

void abcdk_avio_free(AVIOContext **ctx)
{
    AVIOContext *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;

#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58,40,101)
    avio_context_free(&ctx_p);
#else
    if(ctx_p->buffer)
        av_free(ctx_p->buffer);    
    av_free(ctx_p);
#endif
    /* Set NULL(0).*/
    *ctx = NULL;
}

AVIOContext *abcdk_avio_alloc(int buf_blocks, int write_flag, void *opaque)
{
    int buf_size = 8 * 4096; /* 4k bytes 的倍数。 */
    void *buf = NULL;
    AVIOContext *ctx = NULL;

    if (buf_blocks > 0)
        buf_size = buf_blocks * 4096;

    buf = av_malloc(buf_size);
    if (!buf)
        goto final_error;

    ctx = avio_alloc_context((uint8_t *)buf, buf_size, write_flag, NULL, NULL, NULL, NULL);
    if (!ctx)
        goto final_error;

    return ctx;

final_error:

    av_freep(&buf);

    return NULL;
}

void abcdk_avformat_show_options(AVFormatContext *ctx)
{
    if (!ctx)
        return;

    if (ctx->av_class)
        av_opt_show2((void *)&ctx->av_class, NULL, -1, 0);
    else
        av_log(NULL, AV_LOG_INFO, "No options for this.\n");

    if (ctx->iformat)
    {
        if (ctx->iformat->priv_class)
            av_opt_show2((void *)&ctx->iformat->priv_class, NULL, -1, 0);
        else
            av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->iformat->long_name ? ctx->iformat->long_name : ctx->iformat->name));
    }
    if (ctx->oformat)
    {
        if (ctx->oformat->priv_class)
            av_opt_show2((void *)&ctx->oformat->priv_class, NULL, -1, 0);
        else
            av_log(NULL, AV_LOG_INFO, "No options for `%s'.\n", (ctx->oformat->long_name ? ctx->oformat->long_name : ctx->oformat->name));
    }
}

void abcdk_avformat_free(AVFormatContext **ctx)
{
    AVFormatContext *ctx_p = NULL;
    AVIOContext *pb = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;

    /*自定义IO环境不能自动释放，需要单独释放。*/
    if (ctx_p->flags & AVFMT_FLAG_CUSTOM_IO)
        pb = ctx_p->pb;

    if (ctx_p->iformat)
        avformat_close_input(ctx);
    else
        avformat_free_context(ctx_p);

    /*释放自定义IO环境。*/
    if (pb)
        abcdk_avio_free(&pb);

    /* Set NULL(0).*/
    *ctx = NULL;
}

AVFormatContext *abcdk_avformat_input_open(const char *short_name, const char *filename,
                                           AVIOInterruptCB *interrupt_cb, AVIOContext *io_cb,
                                           AVDictionary **dict)
{
    AVInputFormat *fmt = NULL;
    AVFormatContext *ctx = NULL;
    int chk = -1;
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,40,101)
    av_register_all();
#endif 
    avformat_network_init();
    avdevice_register_all();

    assert(filename != NULL);
    
    ctx = avformat_alloc_context();
    if (!ctx)
        return NULL;

    /*如果不知道做什么用的，不要设置这个。*/
    //ctx->flags |= AVFMT_FLAG_NOBUFFER;

    if (interrupt_cb)
        ctx->interrupt_callback = *interrupt_cb;

    if (io_cb)
        ctx->pb = io_cb;

    if (dict)
    {
        /* RTSP默认走TCP，可以减少丢包。*/
        if (strncmp(filename, "rtsp://", 7) == 0)
            av_dict_set(dict, "rtsp_transport", "tcp", 0);

        av_dict_set(dict, "scan_all_pmts", "1", 0);
    }

    fmt = av_find_input_format(short_name);
    chk = avformat_open_input(&ctx, filename, fmt, dict);

    if (chk != 0)
        abcdk_avformat_free(&ctx);

    return ctx;
}

int abcdk_avformat_input_probe(AVFormatContext *ctx, AVDictionary **dict, int dump)
{
    int chk = -1;

    assert(ctx != NULL);
    assert(ctx->iformat != NULL);

    chk = avformat_find_stream_info(ctx, dict);
    if (chk < 0)
        return chk;

    if (dump)
    {
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,40,101)
        av_dump_format(ctx, 0, ctx->filename, 0);
#else
        av_dump_format(ctx, 0, ctx->url, 0);
#endif 
    }

    return chk;
}

int abcdk_avformat_input_read(AVFormatContext *ctx, AVPacket *pkt, enum AVMediaType only_type)
{
    int chk = -1;

    assert(ctx != NULL && pkt != NULL);

    for (;;)
    {
        av_packet_unref(pkt);
        chk = av_read_frame(ctx, pkt);
        if (chk < 0)
            return chk;

        if (only_type < AVMEDIA_TYPE_NB)
        {
            if (ctx->streams[pkt->stream_index]->codec->codec_type == only_type)
                break;
        }
        else
        {
            break;
        }
    }

    return 0;
}

int abcdk_avformat_input_filter(AVFormatContext *ctx, AVPacket *pkt, AVBitStreamFilterContext **filter)
{
    AVBitStreamFilterContext *filter_p = NULL;
    AVCodecContext *codec_p = NULL;
    uint8_t *outbuf = NULL;
    int outbuf_size = 0;
    uint8_t *inbuf = NULL;
    int inbuf_size = 0;
    int chk = -1;

    assert(ctx != NULL && pkt != NULL);

    /*不能保存指针，什么也不做。*/
    if (!filter)
        return 0;

    filter_p = *filter;
    codec_p = ctx->streams[pkt->stream_index]->codec;

    /*如果过滤器未创建，则在内部创建。*/
    if (!filter_p)
    {
        if (codec_p->codec_id == AV_CODEC_ID_H264)
            filter_p = av_bitstream_filter_init("h264_mp4toannexb");
        else if (codec_p->codec_id == AV_CODEC_ID_HEVC)
            filter_p = av_bitstream_filter_init("hevc_mp4toannexb");
        else 
            return 0;

        if(!filter_p)
            return -1;

        /*保存过滤器环境指针。*/
        *filter = filter_p;
    }

    inbuf = pkt->data;
    inbuf_size = pkt->size;
    chk = av_bitstream_filter_filter(filter_p, codec_p, NULL, &outbuf, &outbuf_size, inbuf, inbuf_size,0);// pkt->flags & AV_PKT_FLAG_KEY);
    if (chk < 0)
        return -1;

    /* 
     * 1: 当返回值大于0时，需要释放旧的内存，绑定新的内存。
     * 2: 当返回值等于0时，未创建新的内存，但是数据起始地址可能有变化。
    */
    if (chk > 0)
    {
        av_buffer_unref(&pkt->buf);
        pkt->buf = av_buffer_create(outbuf, outbuf_size, NULL, NULL, 0);
    }
    
    pkt->data = outbuf;
    pkt->size = outbuf_size;

    return 0;
}

AVFormatContext *abcdk_avformat_output_open(const char *short_name, const char *filename, const char *mime_type,
                                            AVIOInterruptCB *interrupt_cb, AVIOContext *io_cb,
                                            AVDictionary **dict)
{
    AVInputFormat *fmt = NULL;
    AVFormatContext *ctx = NULL;
    int chk = -1;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,40,101)
    av_register_all();
#endif 
    avformat_network_init();
    avdevice_register_all();

    assert(filename != NULL);
    
    ctx = avformat_alloc_context();
    if (!ctx)
        return NULL;

    if (interrupt_cb)
        ctx->interrupt_callback = *interrupt_cb;

    if (io_cb)
        ctx->pb = io_cb;

    av_dict_set(&ctx->metadata, "service", "ABCDK",0);
    av_dict_set(&ctx->metadata, "service_name", "ABCDK",0);
    av_dict_set(&ctx->metadata, "service_provider", "ABCDK",0);
    av_dict_set(&ctx->metadata, "artist", "ABCDK",0);

    if (strncmp(filename, "rtsp://", 7) == 0)
        ctx->oformat = av_guess_format("rtsp", NULL, NULL);
    else if (strncmp(filename, "rtmp://", 7) == 0)
        ctx->oformat = av_guess_format("flv", NULL, NULL);

    if (!ctx->oformat)
        ctx->oformat = av_guess_format(short_name, filename, mime_type);

    if (!ctx->oformat)
        goto final_error;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,40,101)
    strncpy(ctx->filename, filename, sizeof(ctx->filename));
#else
    ctx->url = av_strdup(filename);
#endif
    return ctx;

final_error:

    abcdk_avformat_free(&ctx);

    return NULL;
}

AVStream *abcdk_avformat_output_stream(AVFormatContext *ctx, const AVCodec *codec)
{
    assert(ctx != NULL && codec != NULL);
    assert(ctx->oformat);

    return avformat_new_stream(ctx, codec);
}

AVStream *abcdk_avformat_output_stream2(AVFormatContext *ctx, const char *name)
{
    return abcdk_avformat_output_stream(ctx,abcdk_avcodec_find(name,1));
}

AVStream *abcdk_avformat_output_stream3(AVFormatContext *ctx, enum AVCodecID id)
{
    return abcdk_avformat_output_stream(ctx,abcdk_avcodec_find2(id,1));
}

int abcdk_avformat_output_header(AVFormatContext *ctx,AVDictionary **dict,int dump)
{
    int chk = -1;
    assert(ctx != NULL);
    assert(ctx->oformat);

    if ((ctx->oformat->flags & AVFMT_NOFILE) || ctx->pb)
        chk = 0;
    else
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,40,101)
        chk = avio_open(&ctx->pb, ctx->filename, AVIO_FLAG_WRITE);
#else 
        chk = avio_open(&ctx->pb, ctx->url, AVIO_FLAG_WRITE);
#endif
    if (chk != 0)
        return -1;

    if (dict)
    {
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,40,101)
        if (strncmp(ctx->filename, "rtsp://", 7) == 0)
#else
        if (strncmp(ctx->url, "rtsp://", 7) == 0)
#endif
            av_dict_set(dict, "rtsp_transport", "tcp", 0);
    }

    chk = avformat_write_header(ctx, dict);
    if (chk != 0)
        return -1;

    if(dump)
    {
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,40,101)
        av_dump_format(ctx, 0, ctx->filename, 1);
#else
        av_dump_format(ctx, 0, ctx->url, 1);
#endif 
    }

    return 0;
}

int abcdk_avformat_output_write(AVFormatContext *ctx, AVStream *vs, AVPacket *pkt)
{
    assert(ctx != NULL && vs != NULL && pkt != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);
    assert(vs->codec);

    AVRational bq = vs->codec->time_base;
    AVRational cq = vs->time_base;

    if (pkt->pts != (int64_t)AV_NOPTS_VALUE)
        pkt->pts = av_rescale_q(pkt->pts, bq, cq);

    if (pkt->dts != (int64_t)AV_NOPTS_VALUE)
        pkt->dts = av_rescale_q(pkt->dts, bq, cq);

    if (pkt->duration)
        pkt->duration = av_rescale_q(pkt->duration, bq, cq);

    pkt->stream_index = vs->index;

    return av_interleaved_write_frame(ctx, pkt);
}

int abcdk_avformat_output_trailer(AVFormatContext *ctx)
{
    assert(ctx != NULL);

    return av_write_trailer(ctx);
}

int abcdk_avstream_parameters_from_context(AVStream *vs, const AVCodecContext *ctx)
{
    assert(vs != NULL && ctx != NULL);

    /*如果是编码，帧率也一并复制。*/
    if (av_codec_is_encoder(ctx->codec))
    {
        vs->time_base = vs->codec->time_base = ctx->time_base;
        vs->avg_frame_rate = vs->r_frame_rate = av_make_q(ctx->time_base.den, ctx->time_base.num);
    }

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58,35,100)
    avcodec_parameters_from_context(vs->codecpar,ctx);
#else
    vs->codec->codec_type = ctx->codec_type;
    vs->codec->codec_id = ctx->codec_id;
    vs->codec->codec_tag = ctx->codec_tag;
    vs->codec->bit_rate = ctx->bit_rate;
    vs->codec->bits_per_coded_sample = ctx->bits_per_coded_sample;
    vs->codec->bits_per_raw_sample = ctx->bits_per_raw_sample;
    vs->codec->profile = ctx->profile;
    vs->codec->level = ctx->level;
    vs->codec->flags = ctx->flags;

    switch (ctx->codec_type)
    {
    case AVMEDIA_TYPE_VIDEO:
        vs->codec->pix_fmt = ctx->pix_fmt;
        vs->codec->width = ctx->width;
        vs->codec->height = ctx->height;
        vs->codec->gop_size = ctx->gop_size;
        vs->codec->field_order = ctx->field_order;
        vs->codec->color_range = ctx->color_range;
        vs->codec->color_primaries = ctx->color_primaries;
        vs->codec->color_trc = ctx->color_trc;
        vs->codec->colorspace = ctx->colorspace;
        vs->codec->chroma_sample_location = ctx->chroma_sample_location;
        vs->codec->sample_aspect_ratio = ctx->sample_aspect_ratio;
        vs->codec->has_b_frames = ctx->has_b_frames;
        break;
    case AVMEDIA_TYPE_AUDIO:
        vs->codec->sample_fmt = ctx->sample_fmt;
        vs->codec->channel_layout = ctx->channel_layout;
        vs->codec->channels = ctx->channels;
        vs->codec->sample_rate = ctx->sample_rate;
        vs->codec->block_align = ctx->block_align;
        vs->codec->frame_size = ctx->frame_size;
        vs->codec->initial_padding = ctx->initial_padding;
        vs->codec->seek_preroll = ctx->seek_preroll;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        vs->codec->width = ctx->width;
        vs->codec->height = ctx->height;
        break;
    }
    
    if (ctx->extradata)
    {
        vs->codec->extradata_size = 0;
        av_free(vs->codec->extradata);

        vs->codec->extradata = (uint8_t *)av_mallocz(ctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (vs->codec->extradata)
        {
            memcpy(vs->codec->extradata, ctx->extradata, ctx->extradata_size);
            vs->codec->extradata_size = ctx->extradata_size;
        }
        else
        {
            av_log(NULL, AV_LOG_INFO, "@av_mallocz ENOMEM!");
        }
    }

#endif

    return 0;
}

int abcdk_avstream_parameters_to_context(AVCodecContext *ctx, const AVStream *vs)
{
    assert(vs != NULL && ctx != NULL);
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58,35,100)
    avcodec_parameters_to_context(ctx,vs->codecpar);
#else 
    ctx->codec_type = vs->codec->codec_type;
    ctx->codec_id = vs->codec->codec_id;
    ctx->codec_tag = vs->codec->codec_tag;
    ctx->bit_rate = vs->codec->bit_rate;
    ctx->bits_per_coded_sample = vs->codec->bits_per_coded_sample;
    ctx->bits_per_raw_sample = vs->codec->bits_per_raw_sample;
    ctx->profile = vs->codec->profile;
    ctx->level = vs->codec->level;
    ctx->flags = vs->codec->flags;

    switch (vs->codec->codec_type)
    {
    case AVMEDIA_TYPE_VIDEO:
        ctx->pix_fmt = vs->codec->pix_fmt;
        ctx->width = vs->codec->width;
        ctx->height = vs->codec->height;
        ctx->field_order = vs->codec->field_order;
        ctx->color_range = vs->codec->color_range;
        ctx->color_primaries = vs->codec->color_primaries;
        ctx->color_trc = vs->codec->color_trc;
        ctx->colorspace = vs->codec->colorspace;
        ctx->chroma_sample_location = vs->codec->chroma_sample_location;
        ctx->sample_aspect_ratio = vs->codec->sample_aspect_ratio;
        ctx->has_b_frames = vs->codec->has_b_frames;
        break;
    case AVMEDIA_TYPE_AUDIO:
        ctx->sample_fmt = vs->codec->sample_fmt;
        ctx->channel_layout = vs->codec->channel_layout;
        ctx->channels = vs->codec->channels;
        ctx->sample_rate = vs->codec->sample_rate;
        ctx->block_align = vs->codec->block_align;
        ctx->frame_size = vs->codec->frame_size;
        ctx->delay = ctx->initial_padding = vs->codec->initial_padding;
        ctx->seek_preroll = vs->codec->seek_preroll;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        ctx->width = vs->codec->width;
        ctx->height = vs->codec->height;
        break;
    }

    if (vs->codec->extradata)
    {
        ctx->extradata_size = 0;
        av_free(ctx->extradata);
        
        ctx->extradata = (uint8_t *)av_mallocz(vs->codec->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (ctx->extradata)
        {
            memcpy(ctx->extradata, vs->codec->extradata, vs->codec->extradata_size);
            ctx->extradata_size = vs->codec->extradata_size;
        }
        else
        {
            av_log(NULL, AV_LOG_INFO, "@av_mallocz ENOMEM!");
        }
    }
#endif

    return 0;
}

/*-------------Copy from OpenCV----begin------------------*/

#define ABCDK_AVSTREAM_EPS_ZERO 0.000025

double _abcdk_avstream_r2d(AVRational r)
{
    return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
}

double abcdk_avstream_get_duration(AVFormatContext *ctx,AVStream *vs)
{
    double sec = 0.0;

    assert(ctx != NULL && vs != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);

    if (vs->duration > 0)
        sec = (double)vs->duration * _abcdk_avstream_r2d(vs->time_base);

    return sec;
}

double abcdk_avstream_get_fps(AVFormatContext *ctx, AVStream *vs)
{
    double fps = -0.001;

    assert(ctx != NULL && vs != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);

    if (fps < ABCDK_AVSTREAM_EPS_ZERO)
        fps = _abcdk_avstream_r2d(vs->avg_frame_rate);
    if (fps < ABCDK_AVSTREAM_EPS_ZERO)
        fps = _abcdk_avstream_r2d(vs->r_frame_rate);
    if (fps < ABCDK_AVSTREAM_EPS_ZERO)
        fps = 1.0 / _abcdk_avstream_r2d(vs->codec->time_base);

    return fps;
}

double abcdk_avstream_ts2sec(AVFormatContext *ctx,AVStream *vs,int64_t ts)
{
    double sec = -0.000001;

    assert(ctx != NULL && vs != NULL);
    assert(ctx->nb_streams > vs->index && ctx->streams[vs->index] == vs);
        
    sec = (double)(ts - vs->start_time) * _abcdk_avstream_r2d(vs->time_base);
    
    return sec;
}

int64_t abcdk_avstream_ts2num(AVFormatContext *ctx, AVStream *vs,int64_t ts)
{
    int64_t frame_nb = -1;
    double sec = -0.000001;

    sec = abcdk_avstream_ts2sec(ctx, vs, ts);
    if (sec >= 0.0)
        frame_nb = (int64_t)(abcdk_avstream_get_fps(ctx, vs) * sec + 0.5);

    return frame_nb;
}

/*-------------Copy from OpenCV----end------------------*/

/*------------------------------------------------------------------------------------------------*/


#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H && AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H
