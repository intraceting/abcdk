/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/ffmpeg/ffserver.h"

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)


/**流媒体对象。*/
typedef struct _abcdk_ffserver_item
{
    /*配置。*/
    abcdk_ffserver_config_t cfg;

    uint64_t index;

    uint64_t session;
    int closing;
    int64_t open_count;
    volatile int64_t user_active;
    
    abcdk_ffmpeg_config_t ff_cfg;
    abcdk_ffmpeg_t *ff_ctx;

    char src_md5[33];

    int s2d_idx[16];
    int64_t read_key_ns[16];
    int64_t read_gop_ns[16];
    int video_have;

    char record_path_file[PATH_MAX];
    char record_segment_file[PATH_MAX];
    uint64_t record_segment_start;
    uint64_t record_segment_pos[2];

    abcdk_stream_t *live_buf;

    char tip[PATH_MAX];

}abcdk_ffserver_item_t;

/*简单的流媒体服务。*/
struct _abcdk_ffserver
{
    abcdk_tree_t *src_item;
    uint64_t src_session;

    abcdk_tree_t *dst_items;
    abcdk_mutex_t *dst_mutex;

    abcdk_thread_t worker;
    volatile int work_exit;
};//abcdk_ffserver_t;

static int64_t _abcdk_ffserver_clock(uint8_t precision)
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC, precision);
}

static void _abcdk_ffserver_item_destructor_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_ffserver_item_t *ctx_p;

    ctx_p = (abcdk_ffserver_item_t *)obj->pptrs[0];

    abcdk_ffmpeg_destroy(&ctx_p->ff_ctx);
    
    if(ctx_p->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_LIVE)
    {
        if(ctx_p->cfg.u.live.delete_cb)
            ctx_p->cfg.u.live.delete_cb(ctx_p->cfg.u.live.opaque);

        abcdk_stream_destroy(&ctx_p->live_buf);
    }

    abcdk_trace_printf(LOG_INFO, "移除任务(%s)。", ctx_p->tip);
}

static abcdk_tree_t *_abcdk_ffserver_item_alloc(abcdk_ffserver_config_t *cfg)
{
    abcdk_tree_t *ctx;
    abcdk_ffserver_item_t *item_ctx_p;
    
    ctx = abcdk_tree_alloc3(sizeof(abcdk_ffserver_item_t));
    if(!ctx)
        return NULL;

    /*注册析构函数。*/
    abcdk_object_atfree(ctx->obj,_abcdk_ffserver_item_destructor_cb,NULL);

    item_ctx_p = (abcdk_ffserver_item_t *)ctx->obj->pptrs[0];

    item_ctx_p->cfg = *cfg;

    if (item_ctx_p->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_SOURCE)
    {
        /*修复不支持的参数。*/
        item_ctx_p->cfg.u.src.timeout = ABCDK_CLAMP(item_ctx_p->cfg.u.src.timeout, (int)0, (int)5);
        item_ctx_p->cfg.u.src.retry = ABCDK_CLAMP(item_ctx_p->cfg.u.src.retry, (int)1, (int)30);
        item_ctx_p->cfg.u.src.speed = ABCDK_CLAMP(item_ctx_p->cfg.u.src.speed, (float)0.01, (float)100.0);
        item_ctx_p->cfg.u.src.delay_max = ABCDK_CLAMP(item_ctx_p->cfg.u.src.delay_max, (float)0.300, (float)4.999);

        if(item_ctx_p->cfg.tip && !*item_ctx_p->cfg.tip)
            snprintf(item_ctx_p->tip,PATH_MAX,"%s",item_ctx_p->cfg.tip);
        else 
            snprintf(item_ctx_p->tip,PATH_MAX,"%s",item_ctx_p->cfg.u.src.url);
    }
    else if(item_ctx_p->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_RECORD)
    {
        /*修复不支持的参数。*/
        item_ctx_p->cfg.u.record.count = ABCDK_CLAMP(item_ctx_p->cfg.u.record.count,(int)1,(int)65536);
        item_ctx_p->cfg.u.record.duration = ABCDK_CLAMP(item_ctx_p->cfg.u.record.duration,(int)1,(int)3600);

        item_ctx_p->record_segment_pos[0] = UINT64_MAX;

        snprintf(item_ctx_p->record_path_file, PATH_MAX, "%s.mp4.tmp", item_ctx_p->cfg.u.record.prefix);
        snprintf(item_ctx_p->record_segment_file, PATH_MAX, "%s%%llu.mp4", item_ctx_p->cfg.u.record.prefix);

        if(item_ctx_p->cfg.tip && !*item_ctx_p->cfg.tip)
            snprintf(item_ctx_p->tip,PATH_MAX,"%s",item_ctx_p->cfg.tip);
        else 
            snprintf(item_ctx_p->tip,PATH_MAX,"%s",item_ctx_p->record_path_file);
    }
    else if(item_ctx_p->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_PUSH)
    {
        if(item_ctx_p->cfg.tip && !*item_ctx_p->cfg.tip)
            snprintf(item_ctx_p->tip,PATH_MAX,"%s",item_ctx_p->cfg.tip);
        else 
            snprintf(item_ctx_p->tip,PATH_MAX,"%s",item_ctx_p->cfg.u.push.url);
    }
    else if( item_ctx_p->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_LIVE)
    {
        /*修复不支持的参数。*/
        item_ctx_p->cfg.u.live.delay_max = ABCDK_CLAMP(item_ctx_p->cfg.u.live.delay_max,(float)0.300,(float)4.999);

        if(item_ctx_p->cfg.tip && !*item_ctx_p->cfg.tip)
            snprintf(item_ctx_p->tip,PATH_MAX,"%s",item_ctx_p->cfg.tip);
        else 
            snprintf(item_ctx_p->tip,PATH_MAX,"%s","FMP4 Live Streaming");

        /*引用对象。*/
        item_ctx_p->live_buf = abcdk_stream_refer(item_ctx_p->cfg.u.live.buf);
        if(!item_ctx_p->live_buf)
            goto ERR;
    }
    else 
    {
        goto ERR;
    }
    
    item_ctx_p->ff_ctx = NULL;
    item_ctx_p->index = abcdk_sequence_num();
    item_ctx_p->session = 0;
    item_ctx_p->video_have = 0;
    item_ctx_p->open_count = 0;
    abcdk_atomic_store(&item_ctx_p->user_active,_abcdk_ffserver_clock(6));

    return ctx;

ERR:

    abcdk_tree_free(&ctx);
    return NULL;
}

static void _abcdk_ffserver_stop(abcdk_ffserver_t *ctx)
{
    assert(ctx != NULL);

    if(!abcdk_atomic_compare_and_swap(&ctx->work_exit,1,2))
        return;

    abcdk_thread_join(&ctx->worker);
    abcdk_atomic_store(&ctx->work_exit,0);
}

static void *_abcdk_ffserver_worker_routine(void *opaque);

static int _abcdk_ffserver_start(abcdk_ffserver_t *ctx)
{
    int chk;

    if(!abcdk_atomic_compare_and_swap(&ctx->work_exit,0,1))
        return 0;

    ctx->worker.routine = _abcdk_ffserver_worker_routine;
    ctx->worker.opaque = ctx;

    chk = abcdk_thread_create(&ctx->worker,1);
    if(chk != 0)
        goto ERR;

    return 0;

ERR:

    abcdk_atomic_store(&ctx->work_exit,0);
    return -1;
}

void abcdk_ffserver_destroy(abcdk_ffserver_t **ctx)
{
    abcdk_ffserver_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*先停下来。*/
    _abcdk_ffserver_stop(ctx_p);

    abcdk_tree_free(&ctx_p->src_item);
    abcdk_tree_free(&ctx_p->dst_items);
    abcdk_mutex_destroy(&ctx_p->dst_mutex);

    abcdk_heap_free(ctx_p);

}

abcdk_ffserver_t *abcdk_ffserver_create(abcdk_ffserver_config_t *cfg)
{
    abcdk_ffserver_t *ctx;
    int chk;

    assert(cfg != NULL);
    assert(cfg->flag == ABCDK_FFSERVER_CFG_FLAG_SOURCE && cfg->u.src.url != NULL && *cfg->u.src.url != '\0');

    ctx = (abcdk_ffserver_t*)abcdk_heap_alloc(sizeof(abcdk_ffserver_t));
    if(!ctx)
        return NULL;

    ctx->src_item = _abcdk_ffserver_item_alloc(cfg);
    if(!ctx->src_item)
        goto ERR;

    ctx->dst_items = abcdk_tree_alloc3(1);
    if(!ctx->dst_items)
        goto ERR;

    ctx->dst_mutex = abcdk_mutex_create();
    if(!ctx->dst_mutex)
        goto ERR;

    chk = _abcdk_ffserver_start(ctx);
    if(chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_ffserver_destroy(&ctx);
    return NULL;
}


int _abcdk_ffserver_live_write_packet_cb(void *opaque, uint8_t *buf, int buf_size)
{
    abcdk_ffserver_item_t *item_p = (abcdk_ffserver_item_t*)opaque;
    int chk;
    
    chk = abcdk_stream_write_buffer(item_p->live_buf,buf,buf_size);
    if(chk != 0)
        return -1;

    if(item_p->cfg.u.live.ready_cb)
        item_p->cfg.u.live.ready_cb(item_p->cfg.u.live.opaque);

    return buf_size;
}

static int _abcdk_ffserver_dst_init(abcdk_ffserver_t *ctx,abcdk_ffserver_item_t *dst_item)
{
    abcdk_ffserver_item_t *src_item_p;
    AVCodecContext *opt = NULL;
    AVStream *vs_p = NULL, *src_vs_p = NULL;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    AVCodecContext *codecpar = NULL;
#else
    AVCodecParameters *codecpar = NULL;
#endif
    int *idx_p = NULL;
    int chk;

    src_item_p = (abcdk_ffserver_item_t *)ctx->src_item->obj->pptrs[0];

    if (dst_item->session != ctx->src_session || dst_item->closing)
    {
        dst_item->session = ctx->src_session;
        if (dst_item->ff_ctx)
        {
            abcdk_ffmpeg_write_trailer(dst_item->ff_ctx);
            abcdk_ffmpeg_destroy(&dst_item->ff_ctx);
            abcdk_trace_printf(LOG_INFO, "关闭输出环境(%s)。", dst_item->tip);
        }

        if(dst_item->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_RECORD)
        {
            /*录像分段，同时删除较早的录像文件。*/
            abcdk_file_segment(dst_item->record_path_file,dst_item->record_segment_file,dst_item->cfg.u.record.count,1,dst_item->record_segment_pos);
        }
        else if (dst_item->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_LIVE)
        {
            /*直播只能自动重新打开，需要应用层主动创建新行务。*/
            if(dst_item->open_count > 0)
                dst_item->closing = 1;
        }
    }

    /*正在关闭时，直接返回。*/
    if(dst_item->closing)
        return -4;

    /*打开一次即可。*/
    if (dst_item->ff_ctx)
        return 0;

    dst_item->ff_cfg.writer = 1;
    dst_item->ff_cfg.write_flush = 1;

    if(dst_item->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_RECORD)
    {
        dst_item->ff_cfg.file_name = dst_item->record_path_file;
        dst_item->ff_cfg.short_name = "mp4";
    }
    else if(dst_item->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_PUSH)
    {
        dst_item->ff_cfg.file_name = dst_item->cfg.u.push.url;
        dst_item->ff_cfg.short_name = dst_item->cfg.u.push.fmt;
    }
    else if(dst_item->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_LIVE)
    {
        dst_item->ff_cfg.io.opaque = dst_item;
        dst_item->ff_cfg.io.write_cb = _abcdk_ffserver_live_write_packet_cb;
        dst_item->ff_cfg.short_name = "mp4";
    }

    abcdk_trace_printf(LOG_INFO, "创建输出环境(%s)...", dst_item->tip);

    dst_item->ff_ctx = abcdk_ffmpeg_open(&dst_item->ff_cfg);
    if (!dst_item->ff_ctx)
    {
        abcdk_trace_printf(LOG_WARNING, "创建输出环境失败(%s)，稍后重试。",dst_item->tip);
        return -2;
    }

    for (int i = 0; i < abcdk_ffmpeg_streams(src_item_p->ff_ctx); i++)
    {
        src_vs_p = abcdk_ffmpeg_streamptr(src_item_p->ff_ctx, i);

        /*编码器可能未安装，这里初始索引为-1。*/
        idx_p = &dst_item->s2d_idx[src_vs_p->index];
        *idx_p = -1;

        opt = abcdk_avcodec_alloc3(src_vs_p->codecpar->codec_id, 1);
        if (!opt)
            continue;

        /*复制源的编码参数。*/
        abcdk_avstream_parameters_to_context(opt, src_vs_p);

        opt->codec_tag = 0;
        *idx_p = abcdk_ffmpeg_add_stream(dst_item->ff_ctx, opt, 1);
        if (*idx_p < 0)
            goto ERR;

        vs_p = abcdk_ffmpeg_streamptr(dst_item->ff_ctx, *idx_p);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
        codecpar = vs_p->codec;
#else
        codecpar = vs_p->codecpar;
#endif

        /*复制帧率。*/
        vs_p->avg_frame_rate = src_vs_p->avg_frame_rate;
        vs_p->r_frame_rate = src_vs_p->r_frame_rate;

        /*也许没有视频流。*/
        if(codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            dst_item->video_have += 1;

        abcdk_avcodec_free(&opt);
    }

    chk = abcdk_ffmpeg_write_header(dst_item->ff_ctx, 1);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING, "创建输出环境失败(%s)，稍后重试。",dst_item->tip);
        goto ERR;
    }

    /*+1*/
    dst_item->open_count += 1;

    if(dst_item->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_RECORD)
    {
        /*记录开始时间。*/
        dst_item->record_segment_start = _abcdk_ffserver_clock(0);
    }

    return 0;

ERR:

    dst_item->session = 0;
    return -3;
}

static void _abcdk_ffserver_dst_write(abcdk_ffserver_t *ctx,abcdk_ffserver_item_t *dst_item,AVPacket *pkt)
{
    const char *tip_p;
    abcdk_ffserver_item_t *src_item_p;
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

    src_item_p = (abcdk_ffserver_item_t *)ctx->src_item->obj->pptrs[0];

    if(pkt == NULL)
        dst_item->closing = 1;

RECORD_SEGMENT_NEW:

    chk = _abcdk_ffserver_dst_init(ctx,dst_item);
    if (chk != 0)
        return;

    idx_p = &dst_item->s2d_idx[pkt->stream_index];

    /*不支持的跳过。*/
    if (*idx_p < 0)
        return;

    vs_p = abcdk_ffmpeg_streamptr(dst_item->ff_ctx, *idx_p);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    codecpar = vs_p->codec;
#else
    codecpar = vs_p->codecpar;
#endif

    if (dst_item->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_RECORD)
    {
        /*按时间分段。*/
        segment_new = (_abcdk_ffserver_clock(0) - dst_item->record_segment_start > dst_item->cfg.u.record.duration);
        
        /*如果存在视频流，还需要走下面的流程。*/
        if(segment_new && dst_item->video_have)
        {
            /*带视频流的媒体，必须从关键帧开始分段。*/
            if(codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                segment_new = (pkt->flags & AV_PKT_FLAG_KEY);
        }

        /*如果满足分段条件则开始新的分段。*/
        if (segment_new)
        {
            dst_item->session = 0;
            goto RECORD_SEGMENT_NEW;
        }
    }
    else if(dst_item->cfg.flag == ABCDK_FFSERVER_CFG_FLAG_LIVE)
    {
        /*通知缓存数据准备好了。*/
        if(dst_item->cfg.u.live.ready_cb)
            dst_item->cfg.u.live.ready_cb(dst_item->cfg.u.live.opaque);

        /*记录KEY帧和帧分组时间。*/
        if ((pkt->flags & AV_PKT_FLAG_KEY) || (codecpar->codec_type != AVMEDIA_TYPE_VIDEO))
            dst_item->read_key_ns[*idx_p] = dst_item->read_gop_ns[*idx_p] = _abcdk_ffserver_clock(6);

        /*应用层长时间不活动时丢掉一些帧。*/
        delay_ns = (double)(_abcdk_ffserver_clock(6) - abcdk_atomic_load(&dst_item->user_active))/1000000.;
        if(delay_ns > dst_item->cfg.u.live.delay_max)
            dst_item->read_gop_ns[*idx_p] = 0;

        /*可能已经不在同一个GOP中。*/
        if(dst_item->read_key_ns[*idx_p] != dst_item->read_gop_ns[*idx_p])
            obsolete = 1;

        /*按需丢弃延时过多的帧，以便减少延时。*/
        if (obsolete)
        {
            abcdk_trace_printf(LOG_WARNING, "直播(%s)延时超过设定阈值(delay_max=%.3f,delay_ns=%.3f)，丢弃此数据包(index=%d,dts=%.3f,pts=%.3f)。",
                                        dst_item->tip,dst_item->cfg.u.live.delay_max,delay_ns, pkt->stream_index, 
                                        abcdk_ffmpeg_ts2sec(src_item_p->ff_ctx, pkt->stream_index, pkt->dts), 
                                        abcdk_ffmpeg_ts2sec(src_item_p->ff_ctx, pkt->stream_index, pkt->pts));

            return;
        }
    }

    src_vs_p = abcdk_ffmpeg_streamptr(src_item_p->ff_ctx, pkt->stream_index);

    av_init_packet(&pkt_cp);

    /*不能直接使用原始对象，写入前对象的一些参数会被修改。*/
    pkt_cp.data = pkt->data;
    pkt_cp.size = pkt->size;
    pkt_cp.dts = pkt->dts;
    pkt_cp.pts = pkt->pts;
    pkt_cp.duration = pkt->duration;
    pkt_cp.flags = pkt->flags;
    pkt_cp.stream_index = *idx_p;

    chk = abcdk_ffmpeg_write_packet(dst_item->ff_ctx, &pkt_cp, &src_vs_p->time_base);
    if (chk != 0)
        goto ERR;

    return;

ERR:

    dst_item->session = 0;
    return;
}

static void _abcdk_ffserver_write(abcdk_ffserver_t *ctx, AVPacket *pkt)
{
    abcdk_tree_t *task_p = NULL;
    abcdk_object_t *dst_p = NULL; 
    abcdk_ffserver_item_t *dst_item_p = NULL; 

NEXT_ITEM:

    dst_p = NULL; 
    dst_item_p = NULL; 

    abcdk_mutex_lock(ctx->dst_mutex,1);

    if(!task_p)
        task_p = abcdk_tree_child(ctx->dst_items,1);
    else 
        task_p = abcdk_tree_sibling(task_p,0);

    /*增加引用。非常重要，因为任务随时可能被删除。*/
    if(task_p)
        dst_p = abcdk_object_refer(task_p->obj);

    abcdk_mutex_unlock(ctx->dst_mutex);

    /*如果已经到末尾则退出。*/
    if(!dst_p)
        return;

    dst_item_p = (abcdk_ffserver_item_t *)dst_p->pptrs[0];

    _abcdk_ffserver_dst_write(ctx,dst_item_p,pkt);

    /* 减少引用。非常重要，不然无法真的删除任务。*/
    abcdk_object_unref(&dst_p);
 
    goto NEXT_ITEM;
}

int _abcdk_ffserver_src_change_check(abcdk_ffserver_t *ctx)
{
    abcdk_ffserver_item_t *src_item_p;
    AVCodecContext *opt = NULL;
    AVStream *vs_p = NULL;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
    AVCodecContext *codecpar = NULL;
#else
    AVCodecParameters *codecpar = NULL;
#endif
    abcdk_md5_t *md5_ctx;
    char md5_hc[33];
    int chk;

    md5_ctx = abcdk_md5_create();
    if(!md5_ctx)
        return -1;

    src_item_p = (abcdk_ffserver_item_t *)ctx->src_item->obj->pptrs[0];

    abcdk_md5_update(md5_ctx,"ffserver",8);

    /*可能源还未打开。*/
    if(!src_item_p->ff_ctx)
        goto END;

    for (int i = 0; i < abcdk_ffmpeg_streams(src_item_p->ff_ctx); i++)
    {
        vs_p = abcdk_ffmpeg_streamptr(src_item_p->ff_ctx, i);
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 35, 100)
        codecpar = vs_p->codec;
#else
        codecpar = vs_p->codecpar;
#endif
        abcdk_md5_update(md5_ctx,&codecpar->codec_id,sizeof(codecpar->codec_id));
        abcdk_md5_update(md5_ctx,&codecpar->codec_type,sizeof(codecpar->codec_type));
        abcdk_md5_update(md5_ctx,&codecpar->codec_tag,sizeof(codecpar->codec_tag));

        abcdk_trace_printf(LOG_DEBUG, "codec_id=%08x,codec_type=%08x,codec_tag=%08x",
                                    codecpar->codec_id, codecpar->codec_type, codecpar->codec_tag);

        // int fps = abcdk_ffmpeg_fps(src_item_p->ff_ctx,vs_p->index);
        // abcdk_md5_update(md5_ctx,&fps,sizeof(int));

        // abcdk_trace_printf(LOG_DEBUG, "fps=%d",fps);

        if(codecpar->codec_type == AVMEDIA_TYPE_VIDEO || codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE)
        {
            int width = abcdk_ffmpeg_width(src_item_p->ff_ctx,vs_p->index);
            int height = abcdk_ffmpeg_height(src_item_p->ff_ctx,vs_p->index);
            abcdk_md5_update(md5_ctx,&width,sizeof(int));
            abcdk_md5_update(md5_ctx,&height,sizeof(int));

            abcdk_trace_printf(LOG_DEBUG, "width=%d,height=%d",width,height);
        }
        else if(codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
        {
            abcdk_md5_update(md5_ctx,&codecpar->channel_layout,sizeof(codecpar->channel_layout));
            abcdk_md5_update(md5_ctx,&codecpar->channels,sizeof(codecpar->channels));
        }

        if (codecpar->extradata != NULL && codecpar->extradata_size > 0)
            abcdk_md5_update(md5_ctx, codecpar->extradata, codecpar->extradata_size);
    }

END:

    abcdk_md5_final2hex(md5_ctx,md5_hc,0);

    chk = (strncmp(src_item_p->src_md5,md5_hc,32) != 0);
    memcpy(src_item_p->src_md5,md5_hc,32);

    abcdk_md5_destroy(&md5_ctx);

    return chk;
}

void *_abcdk_ffserver_worker_routine(void *opaque)
{
    const char *tip_p;
    abcdk_ffserver_item_t *src_item_p;    
    abcdk_ffserver_t *ctx = NULL;
    AVStream *vs_p = NULL;
    AVPacket pkt = {0};
    uint64_t retry_count = 0;
    int chk;

    ctx = (abcdk_ffserver_t *)opaque;
    src_item_p = (abcdk_ffserver_item_t *)ctx->src_item->obj->pptrs[0];

    /*设置线程名字，日志记录会用到。*/
    abcdk_thread_setname(0, "%x", src_item_p->index);

    src_item_p->ff_cfg.file_name = src_item_p->cfg.u.src.url;
    src_item_p->ff_cfg.short_name = src_item_p->cfg.u.src.fmt;
    src_item_p->ff_cfg.bit_stream_filter = 1;
    src_item_p->ff_cfg.read_speed = src_item_p->cfg.u.src.speed;
    src_item_p->ff_cfg.read_delay_max = src_item_p->cfg.u.src.delay_max;
    src_item_p->ff_cfg.timeout = src_item_p->cfg.u.src.timeout;

    av_init_packet(&pkt);
    
    memset(src_item_p->tip,0,PATH_MAX);
    snprintf(src_item_p->tip,PATH_MAX,"%s",src_item_p->ff_cfg.file_name);
    
RETRY:

    if (!abcdk_atomic_compare(&ctx->work_exit, 1))
        goto END;

    abcdk_ffmpeg_destroy(&src_item_p->ff_ctx);

    /*第一次连接时不需要休息。*/
    if (retry_count++ > 0)
    {
        abcdk_trace_printf(LOG_WARNING, "输入源(%s)已关闭或到末尾，%d秒后重连。", src_item_p->tip, src_item_p->cfg.u.src.retry);
        usleep(src_item_p->cfg.u.src.retry * 1000000);
    }

    abcdk_trace_printf(LOG_INFO, "打开输入源(%s)...", src_item_p->tip);

    src_item_p->ff_ctx = abcdk_ffmpeg_open(&src_item_p->ff_cfg);
    if (!src_item_p->ff_ctx)
        goto RETRY;
    
#if 0
    /* 注：因为ffmpeg接口层还没有实现拼接功能，暂时不能支持按需要更新会话ID的功能。*/
    chk = _abcdk_ffserver_src_change_check(ctx);
    if(chk != 0)
        ctx->src_session = _abcdk_ffserver_clock(6);
#else 
    /*更新会话ID。*/
    ctx->src_session = _abcdk_ffserver_clock(6);
#endif 

LOOP:

    if (!abcdk_atomic_compare(&ctx->work_exit, 1))
        goto END;

    chk = abcdk_ffmpeg_read_packet(src_item_p->ff_ctx, &pkt, -1);
    if (chk < 0)
        goto RETRY;

    vs_p = abcdk_ffmpeg_streamptr(src_item_p->ff_ctx, pkt.stream_index);

    /*修复错误的时长。*/
    if(pkt.duration == 0)
        pkt.duration = av_rescale_q(1, vs_p->time_base, AV_TIME_BASE_Q);

    _abcdk_ffserver_write(ctx, &pkt);
    av_packet_unref(&pkt);

    goto LOOP;

END:

    /*通知关闭连接或文件。*/
    _abcdk_ffserver_write(ctx, NULL);

    av_packet_unref(&pkt);
    abcdk_ffmpeg_destroy(&src_item_p->ff_ctx);

    abcdk_trace_printf(LOG_INFO, "输入源(%s)已关闭。", src_item_p->tip);

    return NULL;
}

void abcdk_ffserver_task_heartbeat(abcdk_ffserver_t *ctx, abcdk_ffserver_task_t *task)
{
    abcdk_tree_t *task_p = NULL;
    abcdk_object_t *dst_p = NULL; 
    abcdk_ffserver_item_t *dst_item_p = NULL; 

    assert(ctx != NULL && task != NULL);

    task_p = (abcdk_tree_t *)task;

    abcdk_mutex_lock(ctx->dst_mutex,1);
    dst_p = abcdk_object_refer(task_p->obj);
    abcdk_mutex_unlock(ctx->dst_mutex);

    dst_item_p = (abcdk_ffserver_item_t*)dst_p->pptrs[0];

    abcdk_atomic_store(&dst_item_p->user_active,_abcdk_ffserver_clock(6));

    abcdk_object_unref(&dst_p);
}

uint64_t abcdk_ffserver_get_index(abcdk_ffserver_t *ctx,abcdk_ffserver_task_t *task)
{
    abcdk_tree_t *task_p = NULL;
    abcdk_object_t *dst_p = NULL; 
    abcdk_ffserver_item_t *dst_item_p = NULL;
    uint64_t index;

    assert(ctx != NULL && task != NULL);

    task_p = (abcdk_tree_t *)task;

    abcdk_mutex_lock(ctx->dst_mutex,1);
    dst_p = abcdk_object_refer(task_p->obj);
    abcdk_mutex_unlock(ctx->dst_mutex);

    dst_item_p = (abcdk_ffserver_item_t*)dst_p->pptrs[0];

    index = dst_item_p->index;

    abcdk_object_unref(&dst_p); 

    return index;
}

void abcdk_ffserver_task_del(abcdk_ffserver_t *ctx, abcdk_ffserver_task_t **task)
{
    abcdk_tree_t *task_p;

    assert(ctx != NULL);

    if(!task || !*task)
        return;

    task_p = (abcdk_tree_t *)*task;
    *task = NULL;

    abcdk_mutex_lock(ctx->dst_mutex,1);
    abcdk_tree_unlink(task_p);
    abcdk_tree_free(&task_p);
    abcdk_mutex_unlock(ctx->dst_mutex);

}

abcdk_ffserver_task_t *abcdk_ffserver_task_add(abcdk_ffserver_t *ctx,abcdk_ffserver_config_t *cfg)
{
    abcdk_tree_t *task;
    abcdk_ffserver_item_t *dst_item;

    assert(ctx != NULL && cfg != NULL);
    assert((cfg->flag == ABCDK_FFSERVER_CFG_FLAG_RECORD && cfg->u.record.prefix != NULL && *cfg->u.record.prefix != '\0') ||
           (cfg->flag == ABCDK_FFSERVER_CFG_FLAG_PUSH && cfg->u.push.url != NULL && *cfg->u.push.url != '\0') ||
           (cfg->flag == ABCDK_FFSERVER_CFG_FLAG_LIVE && cfg->u.live.buf));

    task = _abcdk_ffserver_item_alloc(cfg);
    if(!task)
        return NULL;

    abcdk_mutex_lock(ctx->dst_mutex,1);
    abcdk_tree_insert2(ctx->dst_items,task,0);
    abcdk_mutex_unlock(ctx->dst_mutex);

    return (abcdk_ffserver_task_t *)task;
}


#endif //AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

