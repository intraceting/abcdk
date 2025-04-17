/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/ffrecord.h"

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)


/*简单的视音录像。*/
struct _abcdk_ffrecord
{
    abcdk_ffeditor_config_t ff_cfg;
    abcdk_ffeditor_t *ff_ctx;

    AVCodecContext *codec_ctx[ABCDK_FFMPEG_MAX_STREAMS];
    int index_s2d[ABCDK_FFMPEG_MAX_STREAMS];
    int stream_index;
    int video_have;

    char file_template[PATH_MAX];
    char name_template[NAME_MAX];
    char name_pattern[NAME_MAX];

    int64_t start_number;
    char segment_file[PATH_MAX];
    uint64_t segment_begin;

    const char *save_path;
    const char *segment_prefix;
    int segment_duration;
    int segment_size;

}; // abcdk_ffrecord_t;

void abcdk_ffrecord_destroy(abcdk_ffrecord_t **ctx)
{
    abcdk_ffrecord_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_ffeditor_destroy(&ctx_p->ff_ctx);

    for (int i = 0; i < ABCDK_FFMPEG_MAX_STREAMS; i++)
        abcdk_avcodec_free(&ctx_p->codec_ctx[i]);

    abcdk_heap_free((char*)ctx_p->save_path);
    abcdk_heap_free((char*)ctx_p->segment_prefix);

    abcdk_heap_free(ctx_p);
}

abcdk_ffrecord_t *abcdk_ffrecord_create(const char *save_path, const char *segment_prefix, int segment_duration, int segment_size)
{
    abcdk_ffrecord_t *ctx;

    ctx = (abcdk_ffrecord_t *)abcdk_heap_alloc(sizeof(abcdk_ffrecord_t));
    if (!ctx)
        return NULL;
    
    for (int i = 0; i < ABCDK_FFMPEG_MAX_STREAMS; i++)
        ctx->index_s2d[i] = -1;

    ctx->save_path = abcdk_strdup_safe(save_path);
    ctx->segment_prefix = abcdk_strdup_safe(segment_prefix);
    ctx->segment_duration = segment_duration;
    ctx->segment_size = segment_size;

    /*创建文件名模板。*/
    snprintf(ctx->name_template, NAME_MAX, "%s%%020lld.mp4", (ctx->segment_prefix?ctx->segment_prefix:""));
    snprintf(ctx->name_pattern, NAME_MAX, "%s*.mp4", (ctx->segment_prefix?ctx->segment_prefix:""));
    snprintf(ctx->file_template, PATH_MAX, "%s/%s", ctx->save_path, ctx->name_template);
    

    return ctx;
}

int abcdk_ffrecord_add_stream(abcdk_ffrecord_t *ctx, AVStream *src_stream)
{
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    AVCodecContext *src_codecpar_p = NULL;
#else
    AVCodecParameters *src_codecpar_p = NULL;
#endif

    AVCodecContext *dst_codec_ctx_p = NULL;
    int chk;

    if (src_stream->index >= ABCDK_FFMPEG_MAX_STREAMS)
        return -1;

    dst_codec_ctx_p = ctx->codec_ctx[src_stream->index];
    if (dst_codec_ctx_p)
        return -1; // 重复。

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    src_codecpar_p = src_stream->codec;
#else
    src_codecpar_p = src_stream->codecpar;
#endif

    /*创建编码器。*/
    dst_codec_ctx_p = ctx->codec_ctx[src_stream->index] = abcdk_avcodec_alloc3(src_codecpar_p->codec_id, 1);
    if (!dst_codec_ctx_p)
        return -1;

    /*复制源的编码参数。*/
    chk = abcdk_avstream_parameters_to_context(dst_codec_ctx_p, src_stream);
    if (chk != 0)
        return -1;

    /*视频流统计一下。*/
    if(src_codecpar_p->codec_type == AVMEDIA_TYPE_VIDEO)
        ctx->video_have += 1;

    /*不能复用。*/
    dst_codec_ctx_p->codec_tag = 0;

    return 0;
}

static int64_t _abcdk_ffrecord_find_number(abcdk_ffrecord_t *ctx)
{
    char tmp_name[PATH_MAX] = {0};
    int64_t number_max = 0,tmp_number = 1;
    abcdk_tree_t *dir_ctx = NULL;
    int chk;

    chk = abcdk_dirent_open(&dir_ctx,ctx->save_path);
    if(chk != 0)
        return 0;

    while (1)
    {
        memset(tmp_name,0,PATH_MAX);
        chk = abcdk_dirent_read(dir_ctx, ctx->name_pattern, tmp_name, 0);
        if (chk != 0)
            break;

        chk = sscanf(tmp_name, ctx->name_template , &tmp_number);
        if (chk != 1)
            continue;

        /*取最大的。*/
        number_max = ABCDK_MAX(number_max, tmp_number);
    }

    abcdk_tree_free(&dir_ctx);

    return number_max;
}

static int _abcdk_ffrecord_del_segment(abcdk_ffrecord_t *ctx)
{
    char tmp_file[PATH_MAX] = {0};

    /*关闭已经打开的媒体。*/
    abcdk_ffeditor_destroy(&ctx->ff_ctx);

    /*构造旧的文件名。*/
    snprintf(tmp_file, PATH_MAX, ctx->file_template, ctx->start_number - ctx->segment_size);

    /*删除。*/
    unlink(tmp_file);

    return 0;
}

static int _abcdk_ffrecord_new_segment(abcdk_ffrecord_t *ctx)
{
    int chk;

    if (ctx->start_number <= 0)
    {
        ctx->start_number = _abcdk_ffrecord_find_number(ctx);

        /*删除上次中断时移留的文件。*/
        _abcdk_ffrecord_del_segment(ctx);
    }

    /*关闭前不需要重复打开。*/
    if (ctx->ff_ctx)
        return 0;

    memset(ctx->segment_file, 0, PATH_MAX);
    snprintf(ctx->segment_file, PATH_MAX, ctx->file_template, ++ctx->start_number);

    ctx->ff_cfg.writer = 1;
    ctx->ff_cfg.url = ctx->segment_file;
    ctx->ff_cfg.timeout = 5;
    ctx->ff_cfg.write_flush = 1;

    abcdk_trace_printf(LOG_INFO, TT("打开(%s)..."), ctx->segment_file);

    ctx->ff_ctx = abcdk_ffeditor_open(&ctx->ff_cfg);
    if (!ctx->ff_ctx)
    {
        abcdk_trace_printf(LOG_WARNING, TT("打开(%s)失败，无权限或空间不足。"), ctx->segment_file);
        return -1;
    }

    for (int i = 0; i < ABCDK_FFMPEG_MAX_STREAMS; i++)
    {
        if(!ctx->codec_ctx[i])
            continue;

        chk = ctx->index_s2d[i] = abcdk_ffeditor_add_stream(ctx->ff_ctx, ctx->codec_ctx[i], 1);
        if (chk < 0)
            return -2;
    }

    chk = abcdk_ffeditor_write_header_fmp4(ctx->ff_ctx);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING, TT("写(%s)失败，空间不足或其它。"), ctx->segment_file);
        return -3;
    }

    ctx->segment_begin = abcdk_time_systime(0);

    return 0;
}

static int _abcdk_ffrecord_write_packet(abcdk_ffrecord_t *ctx, AVPacket *src_pkt, AVRational *src_time_base)
{
    AVCodecContext *dst_codec_ctx_p = NULL;
    AVPacket dst_pkt = {0};
    int segment_new = 0;
    int chk;

    assert(ctx != NULL);

    /*强制结束断片。*/
    if (src_pkt == NULL)
    {
        _abcdk_ffrecord_del_segment(ctx);
        return 0;
    }

    if (src_pkt->stream_index >= ABCDK_FFMPEG_MAX_STREAMS)
        return -1;

SEGMENT_NEW:

    chk = _abcdk_ffrecord_new_segment(ctx);
    if (chk != 0)
        return -2;

    /*引用*/
    dst_codec_ctx_p = ctx->codec_ctx[src_pkt->stream_index];

    /*按时间分段。*/
    segment_new = (abcdk_time_systime(0) - ctx->segment_begin > (uint64_t)ctx->segment_duration);

    /*如果存在视频流，还需要走下面的流程。*/
    if (segment_new && ctx->video_have)
    {
        /*带视频流的媒体，必须从关键帧开始分段。*/
        if (dst_codec_ctx_p->codec_type == AVMEDIA_TYPE_VIDEO)
            segment_new = (src_pkt->flags & AV_PKT_FLAG_KEY);
        else 
            segment_new = 0;
    }

    /*如果满足分段条件则开始新的分段。*/
    if (segment_new)
    {
        _abcdk_ffrecord_del_segment(ctx);
        goto SEGMENT_NEW;
    }

    /*初始化。*/
    av_init_packet(&dst_pkt);

    /*不能直接使用原始对象，写入前对象的一些参数会被修改。*/
    dst_pkt.data = src_pkt->data;
    dst_pkt.size = src_pkt->size;
    dst_pkt.dts = src_pkt->dts;
    dst_pkt.pts = src_pkt->pts;
    dst_pkt.duration = src_pkt->duration;
    dst_pkt.flags = src_pkt->flags;
    dst_pkt.stream_index = ctx->index_s2d[src_pkt->stream_index];//从映表中取索引。

    if (src_time_base)
        chk = abcdk_ffeditor_write_packet(ctx->ff_ctx, &dst_pkt, src_time_base);
    else
        chk = abcdk_ffeditor_write_packet2(ctx->ff_ctx, dst_pkt.data, dst_pkt.size, dst_pkt.flags & AV_PKT_FLAG_KEY, dst_pkt.stream_index);
    if (chk != 0)
        return -3;

    return 0;
}

int abcdk_ffrecord_write_packet(abcdk_ffrecord_t *ctx, AVPacket *src_pkt, AVRational *src_time_base)
{
    int chk;

    assert(ctx != NULL);

    chk = _abcdk_ffrecord_write_packet(ctx, src_pkt, src_time_base);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_ffrecord_write_packet2(abcdk_ffrecord_t *ctx, void *data, int size, int keyframe, int stream)
{
    AVPacket src_pkt = {0};
    int chk;

    assert(ctx != NULL && data != NULL && size >0 && stream >= 0);

    /*初始化。*/
    av_init_packet(&src_pkt);

    src_pkt.data = (uint8_t*)data;
    src_pkt.size = size;
    src_pkt.dts = (int64_t)UINT64_C(0x8000000000000000);
    src_pkt.pts = (int64_t)UINT64_C(0x8000000000000000);
    src_pkt.duration = 0;
    src_pkt.flags |= (keyframe?AV_PKT_FLAG_KEY:0);
    src_pkt.stream_index = stream;

    chk = _abcdk_ffrecord_write_packet(ctx,&src_pkt,NULL);
    if(chk != 0)
        return -1;

    return 0;
}

#endif // AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H
