/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/nvr.h"

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

/**NVR任务对象。*/
typedef struct _abcdk_ffmpeg_nvr_item
{
    /*配置。*/
    abcdk_ffmpeg_nvr_config_t cfg;

    /*初始状态。*/
    int init_status;
    
    uint64_t session;
    int closing;
    int64_t open_count;
    uint64_t open_next;

    abcdk_ffeditor_config_t ff_cfg;
    abcdk_ffeditor_t *ff_ctx;

    int s2d_idx[16];
    int64_t read_key_ns[16];
    int64_t read_gop_ns[16];
    int video_have;
    int first_key_ok;

    char record_path_file[PATH_MAX];
    char record_segment_file[PATH_MAX];
    uint64_t record_segment_start;
    uint64_t record_segment_pos[2];

    char tip[PATH_MAX];

} abcdk_ffmpeg_nvr_item_t;

/*简单的流媒体服务。*/
struct _abcdk_ffmpeg_nvr
{
    /**退出标志。0 运行，!0 退出。*/
    volatile int exit_flag;

    /**工作线程。 */
    abcdk_thread_t worker_thread;
    
    /**数据目标。*/
    abcdk_registry_config_t dst_cfg;
    abcdk_registry_t *dst_item;

    /**数据源。 */
    abcdk_ffmpeg_nvr_item_t src_item;

}; // abcdk_ffmpeg_nvr_t;

static uint64_t _abcdk_ffmpeg_nvr_item_key_size_cb(const void *key, void *opaque)
{
    return sizeof(uint64_t);
}

static int _abcdk_ffmpeg_nvr_item_key_compare_cb(const void *key1, const void *key2, void *opaque)
{
    if (ABCDK_PTR2U64(key1, 0) == ABCDK_PTR2U64(key2, 0))
        return 0;
    if (ABCDK_PTR2U64(key1, 0) > ABCDK_PTR2U64(key2, 0))
        return 1;
    if (ABCDK_PTR2U64(key1, 0) < ABCDK_PTR2U64(key2, 0))
        return -1;
}

static void _abcdk_ffmpeg_nvr_item_key_remove_cb(const void *key, abcdk_context_t *userdata, void *opaque)
{
    abcdk_ffmpeg_nvr_item_t *item_ctx_p;

    item_ctx_p = (abcdk_ffmpeg_nvr_item_t *)abcdk_context_get_userdata(userdata);

    abcdk_ffeditor_destroy(&item_ctx_p->ff_ctx);

    abcdk_trace_printf(LOG_INFO, TT("任务(%lld)被删除。"), ABCDK_PTR2U64(key, 0));
}

static int _abcdk_ffmpeg_nvr_item_init(abcdk_ffmpeg_nvr_item_t *item_ctx, abcdk_ffmpeg_nvr_config_t *cfg)
{
    /*清空。*/
    memset(item_ctx, 0, sizeof(*item_ctx));

    item_ctx->cfg = *cfg;
    item_ctx->init_status = 1;

    if (item_ctx->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_SRC)
    {
        /*修复不支持的参数。*/
        item_ctx->cfg.u.src.timeout = ABCDK_CLAMP(item_ctx->cfg.u.src.timeout, (int)0, (int)5);
        item_ctx->cfg.u.src.retry = ABCDK_CLAMP(item_ctx->cfg.u.src.retry, (int)1, (int)30);
        item_ctx->cfg.u.src.speed = ABCDK_CLAMP(item_ctx->cfg.u.src.speed, (float)0.01, (float)100.0);
        item_ctx->cfg.u.src.delay_max = ABCDK_CLAMP(item_ctx->cfg.u.src.delay_max, (float)0.300, (float)4.999);

        if (item_ctx->cfg.tip && !*item_ctx->cfg.tip)
            snprintf(item_ctx->tip, PATH_MAX, "%s", item_ctx->cfg.tip);
        else
            snprintf(item_ctx->tip, PATH_MAX, "%s", item_ctx->cfg.u.src.url);
    }
    else if (item_ctx->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_REC)
    {
        /*修复不支持的参数。*/
        item_ctx->cfg.u.record.count = ABCDK_CLAMP(item_ctx->cfg.u.record.count, (int)1, (int)65536);
        item_ctx->cfg.u.record.duration = ABCDK_CLAMP(item_ctx->cfg.u.record.duration, (int)1, (int)3600);

        item_ctx->record_segment_pos[0] = UINT64_MAX;

        snprintf(item_ctx->record_path_file, PATH_MAX, "%s.mp4.tmp", item_ctx->cfg.u.record.prefix);
        snprintf(item_ctx->record_segment_file, PATH_MAX, "%s%%llu.mp4", item_ctx->cfg.u.record.prefix);

        if (item_ctx->cfg.tip && !*item_ctx->cfg.tip)
            snprintf(item_ctx->tip, PATH_MAX, "%s", item_ctx->cfg.tip);
        else
            snprintf(item_ctx->tip, PATH_MAX, "%s", item_ctx->record_path_file);
    }
    else if (item_ctx->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_PUSH)
    {
        if (item_ctx->cfg.tip && !*item_ctx->cfg.tip)
            snprintf(item_ctx->tip, PATH_MAX, "%s", item_ctx->cfg.tip);
        else
            snprintf(item_ctx->tip, PATH_MAX, "%s", item_ctx->cfg.u.push.url);
    }
    else
    {
        return -1;
    }

    item_ctx->ff_ctx = NULL;
    item_ctx->session = 0;
    item_ctx->video_have = 0;
    item_ctx->open_count = 0;
    item_ctx->open_next = 0;

    return 0;
}

static void *_abcdk_ffmpeg_nvr_worker_thread_routine(void *opaque);

static void _abcdk_ffmpeg_nvr_worker_thread_stop(abcdk_ffmpeg_nvr_t *ctx)
{
    ctx->exit_flag = 1;
    abcdk_thread_join(&ctx->worker_thread);
}

static int _abcdk_ffmpeg_nvr_worker_thread_start(abcdk_ffmpeg_nvr_t *ctx)
{
    int chk;

    ctx->exit_flag = 0;
    ctx->worker_thread.routine = _abcdk_ffmpeg_nvr_worker_thread_routine;
    ctx->worker_thread.opaque = ctx;

    chk = abcdk_thread_create(&ctx->worker_thread, 1);
    if (chk != 0)
        return -1;

    return 0;
}

void abcdk_ffmpeg_nvr_destroy(abcdk_ffmpeg_nvr_t **ctx)
{
    abcdk_ffmpeg_nvr_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*先停下来。*/
    _abcdk_ffmpeg_nvr_worker_thread_stop(ctx_p);

    abcdk_registry_destroy(&ctx_p->dst_item);
    abcdk_heap_free(ctx_p);
}

abcdk_ffmpeg_nvr_t *abcdk_ffmpeg_nvr_create(abcdk_ffmpeg_nvr_config_t *cfg)
{
    abcdk_ffmpeg_nvr_t *ctx;
    int chk;

    assert(cfg != NULL);
    assert(cfg->flag == ABCDK_FFMPEG_NVR_CFG_FLAG_SRC);

    ctx = (abcdk_ffmpeg_nvr_t *)abcdk_heap_alloc(sizeof(abcdk_ffmpeg_nvr_t));
    if (!ctx)
        return NULL;

    ctx->dst_cfg.enable_watch = 1;
    ctx->dst_cfg.opaque = ctx;
    ctx->dst_cfg.key_size_cb = _abcdk_ffmpeg_nvr_item_key_size_cb;
    ctx->dst_cfg.key_compare_cb = _abcdk_ffmpeg_nvr_item_key_compare_cb;
    ctx->dst_cfg.key_remove_cb = _abcdk_ffmpeg_nvr_item_key_remove_cb;
    ctx->dst_item = abcdk_registry_create(&ctx->dst_cfg);
    if (!ctx->dst_item)
        goto ERR;

    chk = _abcdk_ffmpeg_nvr_item_init(&ctx->src_item, cfg);
    if (chk != 0)
        goto ERR;

    chk = _abcdk_ffmpeg_nvr_worker_thread_start(ctx);
    if (chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_ffmpeg_nvr_destroy(&ctx);
    return NULL;
}

void abcdk_ffmpeg_nvr_task_del(abcdk_ffmpeg_nvr_t *ctx, uint64_t id)
{
    assert(ctx != NULL && id != 0);

    abcdk_registry_remove_safe(ctx->dst_item, &id);
}

uint64_t abcdk_ffmpeg_nvr_task_add(abcdk_ffmpeg_nvr_t *ctx, abcdk_ffmpeg_nvr_config_t *cfg)
{
    abcdk_context_t *item_p;
    abcdk_ffmpeg_nvr_item_t *item_ctx_p;
    uint64_t id = abcdk_sequence_num();
    int chk;

    assert(ctx != NULL && cfg != NULL);
    assert(cfg->flag == ABCDK_FFMPEG_NVR_CFG_FLAG_REC || cfg->flag == ABCDK_FFMPEG_NVR_CFG_FLAG_PUSH);

    item_p = abcdk_registry_insert_safe(ctx->dst_item, &id, sizeof(abcdk_ffmpeg_nvr_item_t));
    if (!item_p)
        return -1;

    abcdk_context_wrlock(item_p);

    item_ctx_p = (abcdk_ffmpeg_nvr_item_t *)abcdk_context_get_userdata(item_p);

    if (item_ctx_p->init_status != 0)
    {
        abcdk_trace_printf(LOG_WARNING, TT("任务ID(%llu)已经存在。"), id);
        return abcdk_context_unlock_unref(&item_p, 0);
    }

    chk = _abcdk_ffmpeg_nvr_item_init(item_ctx_p, cfg);
    if (chk != 0)
    {
        abcdk_registry_remove_safe(ctx->dst_item, &id);
        return abcdk_context_unlock_unref(&item_p, 0);
    }

    abcdk_trace_printf(LOG_INFO, TT("任务(%llu)已创建。"), id);

    abcdk_context_unlock_unref(&item_p, 0);

    return id;
}

static int _abcdk_ffmpeg_nvr_dst_init(abcdk_ffmpeg_nvr_t *ctx, abcdk_ffmpeg_nvr_item_t *dst_item)
{
    abcdk_ffmpeg_nvr_item_t *src_item_p;
    AVCodecContext *opt = NULL;
    AVStream *vs_p = NULL, *src_vs_p = NULL;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    AVCodecContext *codecpar = NULL;
#else
    AVCodecParameters *codecpar = NULL;
#endif
    int *idx_p = NULL;
    int chk;

    src_item_p = &ctx->src_item;

    if (dst_item->session != src_item_p->session || dst_item->closing)
    {
        dst_item->session = src_item_p->session;
        if (dst_item->ff_ctx)
        {
            abcdk_ffeditor_write_trailer(dst_item->ff_ctx);
            abcdk_ffeditor_destroy(&dst_item->ff_ctx);
            abcdk_trace_printf(LOG_INFO, TT("关闭输出环境(%s)。"), dst_item->tip);
        }

        if (dst_item->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_REC)
        {
            /*录像分段，同时删除较早的录像文件。*/
            abcdk_file_segment(dst_item->record_path_file, dst_item->record_segment_file, dst_item->cfg.u.record.count, 1, dst_item->record_segment_pos);
        }
        else if(dst_item->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_PUSH)
        {
            /*推流重试不需要太频繁，间隔5秒。*/
            if (dst_item->open_count > 0)
                dst_item->open_next = abcdk_time_systime(0) + 5;
        }
    }

    /*正在关闭时，直接返回。*/
    if (dst_item->closing)
        return -4;

    /*打开一次即可。*/
    if (dst_item->ff_ctx)
        return 0;

    dst_item->ff_cfg.writer = 1;
    dst_item->ff_cfg.write_flush = 1;

    if (dst_item->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_REC)
    {
        dst_item->ff_cfg.file_name = dst_item->record_path_file;
        dst_item->ff_cfg.short_name = "mp4";
    }
    else if (dst_item->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_PUSH)
    {
        dst_item->ff_cfg.file_name = dst_item->cfg.u.push.url;
        dst_item->ff_cfg.short_name = dst_item->cfg.u.push.fmt;

        /*推流重试不需要太频繁。*/
        if (abcdk_time_systime(0) < dst_item->open_next)
            return -15;
    }

    abcdk_trace_printf(LOG_INFO, TT("创建输出环境(%s)..."), dst_item->tip);

    dst_item->ff_ctx = abcdk_ffeditor_open(&dst_item->ff_cfg);
    if (!dst_item->ff_ctx)
    {
        abcdk_trace_printf(LOG_WARNING, TT("创建输出环境失败(%s)，稍后重试。"), dst_item->tip);
        return -2;
    }

    for (int i = 0; i < abcdk_ffeditor_streams(src_item_p->ff_ctx); i++)
    {
        src_vs_p = abcdk_ffeditor_streamptr(src_item_p->ff_ctx, i);

        /*编码器可能未安装，这里初始索引为-1。*/
        idx_p = &dst_item->s2d_idx[src_vs_p->index];
        *idx_p = -1;

        opt = abcdk_avcodec_alloc3(src_vs_p->codecpar->codec_id, 1);
        if (!opt)
            continue;

        /*复制源的编码参数。*/
        abcdk_avstream_parameters_to_context(opt, src_vs_p);

        opt->codec_tag = 0;// 避免格式问题
        *idx_p = abcdk_ffeditor_add_stream(dst_item->ff_ctx, opt, 1);
        if (*idx_p < 0)
            goto ERR;

        vs_p = abcdk_ffeditor_streamptr(dst_item->ff_ctx, *idx_p);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
        codecpar = vs_p->codec;
#else
        codecpar = vs_p->codecpar;
#endif

        /*复制帧率。*/
        vs_p->avg_frame_rate = src_vs_p->avg_frame_rate;
        vs_p->r_frame_rate = src_vs_p->r_frame_rate;

        /*也许没有视频流。*/
        if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            dst_item->video_have += 1;

        abcdk_avcodec_free(&opt);
    }

    chk = abcdk_ffeditor_write_header(dst_item->ff_ctx, 1);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING, TT("创建输出环境失败(%s)，稍后重试。"), dst_item->tip);
        goto ERR;
    }

    /*打开次数累加。*/
    dst_item->open_count += 1;

    /*clear old flag.*/
    dst_item->first_key_ok = 0;

    if (dst_item->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_REC)
    {
        /*记录开始时间。*/
        dst_item->record_segment_start = abcdk_time_systime(0);
    }

    return 0;

ERR:

    dst_item->session = 0;
    return -3;
}

static void _abcdk_ffmpeg_nvr_dst_write(abcdk_ffmpeg_nvr_t *ctx, abcdk_ffmpeg_nvr_item_t *dst_item, AVPacket *pkt)
{
    const char *tip_p;
    abcdk_ffmpeg_nvr_item_t *src_item_p;
    AVStream *vs_p = NULL, *src_vs_p = NULL;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    AVCodecContext *codecpar = NULL;
#else
    AVCodecParameters *codecpar = NULL;
#endif
    AVPacket pkt_cp = {0};
    int *idx_p = NULL;
    int segment_new = 0;
    double delay_ns = 0.;
    int obsolete = 0;
    int chk;

    src_item_p = &ctx->src_item;

    if (pkt == NULL)
        dst_item->closing = 1;

RECORD_SEGMENT_NEW:

    chk = _abcdk_ffmpeg_nvr_dst_init(ctx, dst_item);
    if (chk != 0)
        return;

    idx_p = &dst_item->s2d_idx[pkt->stream_index];

    /*不支持的跳过。*/
    if (*idx_p < 0)
        return;

    vs_p = abcdk_ffeditor_streamptr(dst_item->ff_ctx, *idx_p);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    codecpar = vs_p->codec;
#else
    codecpar = vs_p->codecpar;
#endif
    
    if (!dst_item->first_key_ok)
    {
        if (dst_item->video_have)
        {
            /*视频流，必须从关键帧开始。*/
            if ((codecpar->codec_type == AVMEDIA_TYPE_VIDEO) && (pkt->flags & AV_PKT_FLAG_KEY))
                dst_item->first_key_ok = 1;
        }
        else
        {
            dst_item->first_key_ok = 1;
        }

        if (!dst_item->first_key_ok)
            return;
    }

    if (dst_item->cfg.flag == ABCDK_FFMPEG_NVR_CFG_FLAG_REC)
    {
        /*按时间分段。*/
        segment_new = (abcdk_time_systime(0) - dst_item->record_segment_start > dst_item->cfg.u.record.duration);

        /*如果存在视频流，还需要走下面的流程。*/
        if (segment_new && dst_item->video_have)
        {
            /*带视频流的媒体，必须从关键帧开始分段。*/
            if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                segment_new = (pkt->flags & AV_PKT_FLAG_KEY);
        }

        /*如果满足分段条件则开始新的分段。*/
        if (segment_new)
        {
            dst_item->session = 0;
            goto RECORD_SEGMENT_NEW;
        }
    }

    /**/
    src_vs_p = abcdk_ffeditor_streamptr(src_item_p->ff_ctx, pkt->stream_index);

    av_init_packet(&pkt_cp);

    /*不能直接使用原始对象，写入前对象的一些参数会被修改。*/
    pkt_cp.data = pkt->data;
    pkt_cp.size = pkt->size;
    pkt_cp.dts = pkt->dts;
    pkt_cp.pts = pkt->pts;
    pkt_cp.duration = pkt->duration;
    pkt_cp.flags = pkt->flags;
    pkt_cp.stream_index = *idx_p;

    chk = abcdk_ffeditor_write_packet(dst_item->ff_ctx, &pkt_cp, &src_vs_p->time_base);
    if (chk != 0)
        goto ERR;

    return;

ERR:

    dst_item->session = 0;
    return;
}

static void _abcdk_ffmpeg_nvr_write(abcdk_ffmpeg_nvr_t *ctx, AVPacket *pkt)
{
    void *item_it = NULL;
    abcdk_context_t *item_p = NULL;
    abcdk_ffmpeg_nvr_item_t *item_ctx_p = NULL;

NEXT_ITEM:

    item_p = NULL;
    item_ctx_p = NULL;

    item_p = abcdk_registry_next_safe(ctx->dst_item, &item_it);

    /*如果已经到末尾则退出。*/
    if (!item_p)
        return;

    abcdk_context_wrlock(item_p);

    item_ctx_p = (abcdk_ffmpeg_nvr_item_t *)abcdk_context_get_userdata(item_p);

    _abcdk_ffmpeg_nvr_dst_write(ctx, item_ctx_p, pkt);

    /* 减少引用。非常重要，不然无法真的删除任务。*/
    abcdk_context_unlock_unref(&item_p, 0);

    goto NEXT_ITEM;
}

static void _abcdk_ffmpeg_nvr_process(abcdk_ffmpeg_nvr_t *ctx)
{
    const char *tip_p;
    abcdk_ffmpeg_nvr_item_t *src_item_p;
    AVPacket pkt = {0};
    uint64_t retry_count = 0;
    int chk;

    src_item_p = &ctx->src_item;

    src_item_p->ff_cfg.file_name = src_item_p->cfg.u.src.url;
    src_item_p->ff_cfg.short_name = src_item_p->cfg.u.src.fmt;
    src_item_p->ff_cfg.bit_stream_filter = 0; // 不进解码器，这个不需要。
    src_item_p->ff_cfg.read_speed = src_item_p->cfg.u.src.speed;
    src_item_p->ff_cfg.read_delay_max = src_item_p->cfg.u.src.delay_max;
    src_item_p->ff_cfg.timeout = src_item_p->cfg.u.src.timeout;

    av_init_packet(&pkt);

    memset(src_item_p->tip, 0, PATH_MAX);
    snprintf(src_item_p->tip, PATH_MAX, "%s", src_item_p->ff_cfg.file_name);

RETRY:

    if (ctx->exit_flag)
        goto END;

    abcdk_ffeditor_destroy(&src_item_p->ff_ctx);

    /*第一次连接时不需要休息。*/
    if (retry_count++ > 0)
    {
        /*本地文件不需要重试。*/
        if (access(src_item_p->cfg.u.src.url, F_OK) == 0)
            goto END;

        abcdk_trace_printf(LOG_WARNING, TT("输入源(%s)已关闭或到末尾，%d秒后重连。"), src_item_p->tip, src_item_p->cfg.u.src.retry);
        usleep(src_item_p->cfg.u.src.retry * 1000000);
    }

    abcdk_trace_printf(LOG_INFO, TT("打开输入源(%s)..."), src_item_p->tip);

    src_item_p->ff_ctx = abcdk_ffeditor_open(&src_item_p->ff_cfg);
    if (!src_item_p->ff_ctx)
        goto RETRY;

    /*更新会话ID。*/
    src_item_p->session = abcdk_time_systime(6);

LOOP:

    if (ctx->exit_flag)
        goto END;

    chk = abcdk_ffeditor_read_packet(src_item_p->ff_ctx, &pkt, -1);
    if (chk < 0)
        goto RETRY;

    /**/
    _abcdk_ffmpeg_nvr_write(ctx, &pkt);
    av_packet_unref(&pkt);

    goto LOOP;

END:

    /*通知关闭连接或文件。*/
    _abcdk_ffmpeg_nvr_write(ctx, NULL);

    /**/
    abcdk_ffeditor_destroy(&src_item_p->ff_ctx);
    av_packet_unref(&pkt);

    abcdk_trace_printf(LOG_INFO, TT("输入源(%s)已关闭。"), src_item_p->tip);

    return;
}

static void *_abcdk_ffmpeg_nvr_worker_thread_routine(void *opaque)
{
    abcdk_ffmpeg_nvr_t *ctx = (abcdk_ffmpeg_nvr_t *)opaque;

    /*设置线程名字，日志记录会用到。*/
    abcdk_thread_setname(0, "%x", abcdk_sequence_num());

    _abcdk_ffmpeg_nvr_process(ctx);

    return NULL;
}

#else // AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

void abcdk_ffmpeg_nvr_destroy(abcdk_ffmpeg_nvr_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
}

abcdk_ffmpeg_nvr_t *abcdk_ffmpeg_nvr_create(abcdk_ffmpeg_nvr_config_t *cfg)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
}

void abcdk_ffmpeg_nvr_task_del(abcdk_ffmpeg_nvr_t *ctx, abcdk_ffmpeg_nvr_t **task)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
}

abcdk_ffmpeg_nvr_t *abcdk_ffmpeg_nvr_task_add(abcdk_ffmpeg_nvr_t *ctx, abcdk_ffmpeg_nvr_config_t *cfg)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFMPEG工具。"));
    return NULL;
}

#endif // AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H
