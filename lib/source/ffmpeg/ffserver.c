/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/ffmpeg/ffserver.h"


/**流媒体对象。*/
typedef struct _abcdk_ffserver_item
{
    void *father;

    int id;

    abcdk_ffmpeg_config_t ff_cfg;
    abcdk_ffmpeg_t *ff_ctx;

    int reopen;
    int s2d_idx[16];

    int64_t read_key_ns[16];
    int64_t read_gop_ns[16];

    uint64_t segment_pos[2];
}abcdk_ffserver_item_t;

/*简单的流媒体服务。*/
struct _abcdk_ffserver
{
    /*配置。*/
    abcdk_ffserver_config_t cfg;
    
    abcdk_ffserver_item_t src;

    abcdk_ffserver_item_t record;
    char record_path_file[PATH_MAX];
    char record_segment_file[PATH_MAX];
    uint64_t record_segment_start;

    abcdk_ffserver_item_t push;

    abcdk_object_t *live_list;
    abcdk_object_t *live_ids;
    abcdk_mutex_t *live_mutex;

    abcdk_thread_t worker;
    volatile int work_exit;
};//abcdk_ffserver_t;

static int64_t _abcdk_ffserver_clock(uint8_t precision)
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC, precision);
}

void abcdk_ffserver_destroy(abcdk_ffserver_t **ctx)
{
    abcdk_ffserver_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*先停下来。*/
    abcdk_ffserver_stop(ctx_p);

    abcdk_object_unref(&ctx_p->live_list);
    abcdk_object_unref(&ctx_p->live_ids);
    abcdk_mutex_destroy(&ctx_p->live_mutex);

    abcdk_heap_free(ctx_p);

}

static int _abcdk_ffserver_init(abcdk_ffserver_t *ctx)
{
    ctx->live_list = abcdk_object_alloc3(sizeof(abcdk_ffserver_item_t),ctx->cfg.live_count_max);
    if(!ctx->live_list)
        return -1;

    ctx->live_ids = abcdk_object_alloc2(abcdk_align(ctx->cfg.live_count_max,2));
    if(!ctx->live_ids)
        return -2;

    ctx->live_mutex = abcdk_mutex_create();
    if(!ctx->live_mutex)
        return -2;

    return 0;
}

abcdk_ffserver_t *abcdk_ffserver_create(abcdk_ffserver_config_t *cfg)
{
    abcdk_ffserver_t *ctx;
    int chk;

    assert(cfg != NULL);

    ctx = (abcdk_ffserver_t*)abcdk_heap_alloc(sizeof(abcdk_ffserver_t));
    if(!ctx)
        return NULL;

    ctx->cfg = *cfg;

    /*修复不支持的参数。*/
    ctx->cfg.src_timeout = ABCDK_CLAMP(ctx->cfg.src_timeout,(int)-1,(int)5);
    ctx->cfg.src_retry = ABCDK_CLAMP(ctx->cfg.src_retry,(int)1,(int)30);
    ctx->cfg.src_speed = ABCDK_CLAMP(ctx->cfg.src_speed,(float)0.01,(float)100.0);
    ctx->cfg.src_delay_max = ABCDK_CLAMP(ctx->cfg.src_delay_max,(float)0.300,(float)4.999);
    ctx->cfg.record_count = ABCDK_CLAMP(ctx->cfg.record_count,(int)1,(int)65536);
    ctx->cfg.record_duration = ABCDK_CLAMP(ctx->cfg.record_duration,(int)1,(int)3600);
    ctx->cfg.live_delay_max = ABCDK_CLAMP(ctx->cfg.live_delay_max,(float)0.300,(float)4.999);
    ctx->cfg.live_count_max = ABCDK_CLAMP(ctx->cfg.live_count_max,(int)1,(int)99999);

    chk = _abcdk_ffserver_init(ctx);
    if(chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_ffserver_destroy(&ctx);
    return NULL;
}

void abcdk_ffserver_stop(abcdk_ffserver_t *ctx)
{
    assert(ctx != NULL);

    if(!abcdk_atomic_compare_and_swap(&ctx->work_exit,1,2))
        return;

    abcdk_thread_join(&ctx->worker);
    abcdk_atomic_store(&ctx->work_exit,0);
}

static void *_abcdk_ffserver_worker_routine(void *opaque);

int abcdk_ffserver_start(abcdk_ffserver_t *ctx)
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

static int _abcdk_ffserver_record_init(abcdk_ffserver_t *ctx,int closing)
{
    AVCodecContext *opt = NULL;
    AVStream *av_p = NULL, *src_vs_p = NULL;
    int *idx_p = NULL;
    int chk;

    if(!ctx->cfg.record_prefix || !*ctx->cfg.record_prefix)
        return -1;

    if(!ctx->record_path_file[0])
    {
        snprintf(ctx->record_path_file,PATH_MAX,"%srecord.tmp",ctx->cfg.record_prefix);
        snprintf(ctx->record_segment_file,PATH_MAX,"%s%%llu.mp4",ctx->cfg.record_prefix);
    }
    
    if (ctx->record.reopen)
    {
        ctx->record.reopen = 0;
        if (ctx->record.ff_ctx)
            abcdk_ffmpeg_write_trailer(ctx->record.ff_ctx);

        abcdk_ffmpeg_destroy(&ctx->record.ff_ctx);

        /*删除过多的分段录像文件。*/
        abcdk_file_segment(ctx->record_path_file,ctx->record_segment_file,ctx->cfg.record_count,1,ctx->record.segment_pos);
    }

    /*正在关闭时，直接返回。*/
    if(closing)
        return -1;

    /*打开一次即可。*/
    if (ctx->record.ff_ctx)
        return 0;

    ctx->record.ff_cfg.writer = 1;
    ctx->record.ff_cfg.file_name = ctx->record_path_file;
    ctx->record.ff_cfg.short_name = "mp4";
    ctx->record.ff_cfg.write_flush = 1;

    abcdk_trace_output(LOG_INFO, "打开录像文件(%s)...", ctx->record_path_file);

    ctx->record.ff_ctx = abcdk_ffmpeg_open(&ctx->record.ff_cfg);
    if (!ctx->record.ff_ctx)
    {
        abcdk_trace_output(LOG_WARNING, "打开录像文件(%s)失败，稍后重试。", ctx->record_path_file);
        return -2;
    }

    for (int i = 0; i < abcdk_ffmpeg_streams(ctx->src.ff_ctx); i++)
    {
        src_vs_p = abcdk_ffmpeg_streamptr(ctx->src.ff_ctx, i);

        /*编码器可能未安装，这里初始索引为-1。*/
        idx_p = &ctx->record.s2d_idx[src_vs_p->index];
        *idx_p = -1;

        opt = abcdk_avcodec_alloc3(src_vs_p->codecpar->codec_id, 1);
        if (!opt)
            continue;

        /*复制源的编码参数。*/
        abcdk_avstream_parameters_to_context(opt, src_vs_p);

        opt->codec_tag = 0;
        *idx_p = abcdk_ffmpeg_add_stream(ctx->record.ff_ctx, opt, 1);
        if (*idx_p < 0)
            goto ERR;

        av_p = abcdk_ffmpeg_streamptr(ctx->record.ff_ctx, *idx_p);

        /*复制帧率。*/
        av_p->avg_frame_rate = src_vs_p->avg_frame_rate;
        av_p->r_frame_rate = src_vs_p->r_frame_rate;

        abcdk_avcodec_free(&opt);
    }

    chk = abcdk_ffmpeg_write_header(ctx->record.ff_ctx, 1);
    if (chk != 0)
        goto ERR;

    /*记录开始时间。*/
    ctx->record_segment_start = _abcdk_ffserver_clock(0);

    return 0;

ERR:

    ctx->record.reopen = 1;
    return -3;
}

static void _abcdk_ffserver_recored(abcdk_ffserver_t *ctx,AVPacket *pkt)
{
    AVStream *av_p = NULL, *src_vs_p = NULL;
    AVPacket pkt_cp = {0};
    int *idx_p = NULL;
    int chk;

SEGMENT_NEW:

    chk = _abcdk_ffserver_record_init(ctx,pkt == NULL);
    if (chk != 0)
        return;
    
    if(!pkt)
        return;

    idx_p = &ctx->record.s2d_idx[pkt->stream_index];

    /*不支持的跳过。*/
    if (*idx_p < 0)
        return;

    /*如果达到分段要求，并且当前帧还是关键帧，则开始新的分段。*/
    if ((_abcdk_ffserver_clock(0) - ctx->record_segment_start > ctx->cfg.record_duration) && pkt->flags & AV_PKT_FLAG_KEY)
    {
        ctx->record.reopen = 1;
        goto SEGMENT_NEW;
    }

    src_vs_p = abcdk_ffmpeg_streamptr(ctx->src.ff_ctx, pkt->stream_index);

    av_init_packet(&pkt_cp);

    /*不能直接使用原始对象，写入前对象的一些参数会被修改。*/
    pkt_cp.data = pkt->data;
    pkt_cp.size = pkt->size;
    pkt_cp.dts = pkt->dts;
    pkt_cp.pts = pkt->pts;
    pkt_cp.duration = pkt->duration;
    pkt_cp.flags = pkt->flags;
    pkt_cp.stream_index = *idx_p;

    chk = abcdk_ffmpeg_write_packet(ctx->record.ff_ctx, &pkt_cp, &src_vs_p->time_base);
    if (chk != 0)
        goto ERR;

    return;

ERR:

    ctx->record.reopen = 1;
    return;
}

static int _abcdk_ffserver_push_init(abcdk_ffserver_t *ctx,int closing)
{
    AVCodecContext *opt = NULL;
    AVStream *av_p = NULL, *src_vs_p = NULL;
    int *idx_p = NULL;
    int chk;

    if(!ctx->cfg.push_url || !*ctx->cfg.push_url)
        return -1;
    
    if (ctx->push.reopen)
    {
        ctx->push.reopen = 0;
        if (ctx->push.ff_ctx)
            abcdk_ffmpeg_write_trailer(ctx->push.ff_ctx);

        abcdk_ffmpeg_destroy(&ctx->push.ff_ctx);
    }

    /*正在关闭时，直接返回。*/
    if(closing)
        return -4;

    /*打开一次即可。*/
    if (ctx->push.ff_ctx)
        return 0;

    ctx->push.ff_cfg.writer = 1;
    ctx->push.ff_cfg.file_name = ctx->cfg.push_url;
    ctx->push.ff_cfg.short_name = ctx->cfg.push_fmt;
    ctx->push.ff_cfg.write_flush = 1;

    abcdk_trace_output(LOG_INFO, "连接推流地址(%s)...", ctx->cfg.push_url);

    ctx->push.ff_ctx = abcdk_ffmpeg_open(&ctx->push.ff_cfg);
    if (!ctx->push.ff_ctx)
    {
        abcdk_trace_output(LOG_WARNING, "连接推流地址(%s)失败，稍后重试。", ctx->cfg.push_url);
        return -2;
    }

    for (int i = 0; i < abcdk_ffmpeg_streams(ctx->src.ff_ctx); i++)
    {
        src_vs_p = abcdk_ffmpeg_streamptr(ctx->src.ff_ctx, i);

        /*编码器可能未安装，这里初始索引为-1。*/
        idx_p = &ctx->push.s2d_idx[src_vs_p->index];
        *idx_p = -1;

        opt = abcdk_avcodec_alloc3(src_vs_p->codecpar->codec_id, 1);
        if (!opt)
            continue;

        /*复制源的编码参数。*/
        abcdk_avstream_parameters_to_context(opt, src_vs_p);

        opt->codec_tag = 0;
        *idx_p = abcdk_ffmpeg_add_stream(ctx->push.ff_ctx, opt, 1);
        if (*idx_p < 0)
            goto ERR;

        av_p = abcdk_ffmpeg_streamptr(ctx->push.ff_ctx, *idx_p);

        /*复制帧率。*/
        av_p->avg_frame_rate = src_vs_p->avg_frame_rate;
        av_p->r_frame_rate = src_vs_p->r_frame_rate;

        abcdk_avcodec_free(&opt);
    }

    chk = abcdk_ffmpeg_write_header(ctx->push.ff_ctx, 1);
    if (chk != 0)
        goto ERR;

    return 0;

ERR:

    ctx->record.reopen = 1;
    return -3;
}

static void _abcdk_ffserver_push(abcdk_ffserver_t *ctx, AVPacket *pkt)
{
    AVStream *av_p = NULL, *src_vs_p = NULL;
    AVPacket pkt_cp = {0};
    int *idx_p = NULL;
    int chk;

    chk = _abcdk_ffserver_push_init(ctx, pkt == NULL);
    if (chk != 0)
        return;

    idx_p = &ctx->push.s2d_idx[pkt->stream_index];

    /*不支持的跳过。*/
    if (*idx_p < 0)
        return;

    src_vs_p = abcdk_ffmpeg_streamptr(ctx->src.ff_ctx, pkt->stream_index);

    av_init_packet(&pkt_cp);

    /*不能直接使用原始对象，写入前对象的一些参数会被修改。*/
    pkt_cp.data = pkt->data;
    pkt_cp.size = pkt->size;
    pkt_cp.dts = pkt->dts;
    pkt_cp.pts = pkt->pts;
    pkt_cp.duration = pkt->duration;
    pkt_cp.flags = pkt->flags;
    pkt_cp.stream_index = *idx_p;

    chk = abcdk_ffmpeg_write_packet(ctx->push.ff_ctx, &pkt_cp, &src_vs_p->time_base);
    if (chk != 0)
        goto ERR;

    return;

ERR:

    ctx->push.reopen = 1;
    return;
}


int _abcdk_ffserver_live_write_packet_cb(void *opaque, uint8_t *buf, int buf_size)
{
    abcdk_ffserver_item_t *item_p = (abcdk_ffserver_item_t*)opaque;
    abcdk_ffserver_t *ctx_p = (abcdk_ffserver_t*)item_p->father;
    
    ctx_p->cfg.live_cb(ctx_p->cfg.live_opaque,item_p->id,buf,buf_size);

    return buf_size;
}

static int _abcdk_ffserver_live_init(abcdk_ffserver_t *ctx,abcdk_ffserver_item_t *item_ctx,int closing)
{
    AVCodecContext *opt = NULL;
    AVStream *av_p = NULL, *src_vs_p = NULL;
    int *idx_p = NULL;
    int chk;

    if(!ctx->cfg.live_cb)
        return -1;
    
    if (item_ctx->reopen)
    {
        item_ctx->reopen = 0;
        if (item_ctx->ff_ctx)
            abcdk_ffmpeg_write_trailer(item_ctx->ff_ctx);

        abcdk_ffmpeg_destroy(&item_ctx->ff_ctx);
    }

    /*正在关闭时，直接返回。*/
    if(closing)
        return -4;

    /*打开一次即可。*/
    if (item_ctx->ff_ctx)
        return 0;

    item_ctx->ff_cfg.writer = 1;
    item_ctx->ff_cfg.io.opaque = item_ctx;
    item_ctx->ff_cfg.io.write_cb = _abcdk_ffserver_live_write_packet_cb;
    item_ctx->ff_cfg.short_name = "mp4";
    item_ctx->ff_cfg.write_flush = 1;

    abcdk_trace_output(LOG_INFO, "创建直播环境...");

    item_ctx->ff_ctx = abcdk_ffmpeg_open(&item_ctx->ff_cfg);
    if (!item_ctx->ff_ctx)
    {
        abcdk_trace_output(LOG_WARNING, "创建直播环境失败，稍后重试。");
        return -2;
    }

    for (int i = 0; i < abcdk_ffmpeg_streams(ctx->src.ff_ctx); i++)
    {
        src_vs_p = abcdk_ffmpeg_streamptr(ctx->src.ff_ctx, i);

        /*编码器可能未安装，这里初始索引为-1。*/
        idx_p = &item_ctx->s2d_idx[src_vs_p->index];
        *idx_p = -1;

        opt = abcdk_avcodec_alloc3(src_vs_p->codecpar->codec_id, 1);
        if (!opt)
            continue;

        /*复制源的编码参数。*/
        abcdk_avstream_parameters_to_context(opt, src_vs_p);

        opt->codec_tag = 0;
        *idx_p = abcdk_ffmpeg_add_stream(item_ctx->ff_ctx, opt, 1);
        if (*idx_p < 0)
            goto ERR;

        av_p = abcdk_ffmpeg_streamptr(item_ctx->ff_ctx, *idx_p);

        /*复制帧率。*/
        av_p->avg_frame_rate = src_vs_p->avg_frame_rate;
        av_p->r_frame_rate = src_vs_p->r_frame_rate;

        abcdk_avcodec_free(&opt);
    }

    chk = abcdk_ffmpeg_write_header(item_ctx->ff_ctx, 1);
    if (chk != 0)
        goto ERR;

    return 0;

ERR:

    ctx->record.reopen = 1;
    return -3;
}

static void _abcdk_ffserver_live_write(abcdk_ffserver_t *ctx,abcdk_ffserver_item_t *item_ctx,AVPacket *pkt)
{
    AVStream *av_p = NULL, *src_vs_p = NULL;
    AVPacket pkt_cp = {0};
    int *idx_p = NULL;
    int chk;

    chk = _abcdk_ffserver_live_init(ctx,item_ctx,pkt == NULL);
    if (chk != 0)
        return;

    idx_p = &item_ctx->s2d_idx[pkt->stream_index];

    /*不支持的跳过。*/
    if (*idx_p < 0)
        return;

    src_vs_p = abcdk_ffmpeg_streamptr(ctx->src.ff_ctx, pkt->stream_index);

    av_init_packet(&pkt_cp);

    /*不能直接使用原始对象，写入前对象的一些参数会被修改。*/
    pkt_cp.data = pkt->data;
    pkt_cp.size = pkt->size;
    pkt_cp.dts = pkt->dts;
    pkt_cp.pts = pkt->pts;
    pkt_cp.duration = pkt->duration;
    pkt_cp.flags = pkt->flags;
    pkt_cp.stream_index = *idx_p;

    chk = abcdk_ffmpeg_write_packet(item_ctx->ff_ctx, &pkt_cp, &src_vs_p->time_base);
    if (chk != 0)
        goto ERR;

    return;

ERR:

    item_ctx->reopen = 1;
    return;
}

static void _abcdk_ffserver_live(abcdk_ffserver_t *ctx, AVPacket *pkt)
{
    abcdk_ffserver_item_t *item_p = NULL;
    int id = 0;
    int open_chk;
    int chk;

NEXT_ITEM:

    id += 1;
    if (id > ctx->cfg.live_count_max)
        return;

    abcdk_mutex_lock(ctx->live_mutex,1);
    open_chk = abcdk_bloom_filter(ctx->live_ids->pptrs[0], ctx->live_ids->sizes[0], id);
    abcdk_mutex_unlock(ctx->live_mutex);

    item_p = (abcdk_ffserver_item_t *)ctx->live_list->pptrs[id-1];
    item_p->father = ctx;
    item_p->id = id;
    item_p->reopen = !open_chk;//如果应用层主动关闭直播，这里通知关闭。

    _abcdk_ffserver_live_write(ctx,item_p,open_chk?pkt:NULL);

    goto NEXT_ITEM;
}

static void _abcdk_ffserver_write(abcdk_ffserver_t *ctx, AVPacket *pkt)
{
    for (int i = 0; i < 3; i++)
    {
        if(i == 0)
            _abcdk_ffserver_recored(ctx, pkt);
        else if(i == 1)
            _abcdk_ffserver_push(ctx, pkt);
        else if(i == 2)
            _abcdk_ffserver_live(ctx, pkt);
    }
}

void *_abcdk_ffserver_worker_routine(void *opaque)
{
    abcdk_ffserver_t *ctx = NULL;
    AVStream *vs_p = NULL;
    AVPacket pkt = {0};
    uint64_t retry_count = 0;
    int chk;

    ctx = (abcdk_ffserver_t *)opaque;

    /*告知录像接续进行。*/
    ctx->record.segment_pos[0] = UINT64_MAX;

    ctx->src.ff_cfg.file_name = ctx->cfg.src_url;
    ctx->src.ff_cfg.short_name = ctx->cfg.src_fmt;
    ctx->src.ff_cfg.bit_stream_filter = 1;
    ctx->src.ff_cfg.read_speed = ctx->cfg.src_speed;
    ctx->src.ff_cfg.read_delay_max = ctx->cfg.src_delay_max;
    ctx->src.ff_cfg.timeout = ctx->cfg.src_timeout;

RETRY:

    if (!abcdk_atomic_compare(&ctx->work_exit, 1))
        goto END;

    abcdk_ffmpeg_destroy(&ctx->src.ff_ctx);

    /*通知断开重连。*/
    ctx->record.reopen = 1;
    ctx->push.reopen = 1;

    /*第一次连接时不需要休息。*/
    if (retry_count++ > 0)
    {
        abcdk_trace_output(LOG_WARNING, "源地址(%s)已关闭或到末尾，%d秒后重连。", ctx->cfg.src_url, ctx->cfg.src_retry);
        usleep(ctx->cfg.src_retry * 1000000);
    }

    abcdk_trace_output(LOG_INFO, "连接源地址(%s)...", ctx->cfg.src_url);

    ctx->src.ff_ctx = abcdk_ffmpeg_open(&ctx->src.ff_cfg);
    if (!ctx->src.ff_ctx)
        goto RETRY;

LOOP:

    if (!abcdk_atomic_compare(&ctx->work_exit, 1))
        goto END;

    abcdk_ffmpeg_read_delay(ctx->src.ff_ctx, 0);

    av_init_packet(&pkt);
    chk = abcdk_ffmpeg_read_packet(ctx->src.ff_ctx, &pkt, -1);
    if (chk < 0)
        goto RETRY;

    vs_p = abcdk_ffmpeg_streamptr(ctx->src.ff_ctx, pkt.stream_index);

    /*修复错误的时长。*/
    if(pkt.duration == 0)
        pkt.duration = av_rescale_q(1, vs_p->time_base, AV_TIME_BASE_Q);

//    abcdk_trace_output(LOG_DEBUG,"pts=%lld,dts=%lld",pkt.pts,pkt.dts);

    _abcdk_ffserver_write(ctx, &pkt);

    goto LOOP;

END:

    /*通知断开重连。*/
    ctx->record.reopen = 1;
    ctx->push.reopen = 1;

    /*通知关闭连接或文件。*/
    _abcdk_ffserver_write(ctx, NULL);

    av_packet_unref(&pkt);
    abcdk_ffmpeg_destroy(&ctx->src.ff_ctx);

    return NULL;
}