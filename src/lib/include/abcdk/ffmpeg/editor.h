/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_EDITOR_H
#define ABCDK_FFMPEG_EDITOR_H

#include "abcdk/util/trace.h"
#include "abcdk/util/time.h"
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/util.h"
#include "abcdk/ffmpeg/encoder.h"
#include "abcdk/ffmpeg/decoder.h"
#include "abcdk/ffmpeg/bsf.h"


__BEGIN_DECLS

typedef struct _abcdk_ffmpeg_editor_param
{
    /**格式.*/
    const char *fmt;

    /**地址. */
    const char *url;

    /**虚拟的输入/输出. */
    struct
    {
        int (*read_cb)(void *opaque, uint8_t *buf, int size);
        int (*write_cb)(void *opaque, uint8_t *buf, int size);
        void *opaque;
    } vio;

    /**超时(秒). */
    int timeout;

    /**
     * RTSP传输协议. 
     * 
     * 0: auto.
     * 1: udp.
     * 2: tcp.
    */
    int rtsp_transport;

    /**是否启用编解码器.*/
    int coder_enable;

    /**读,是否忽略视频. */
    int read_ignore_video;

    /**读,是否忽略音频. */
    int read_ignore_audio;

    /**读,是否忽略字幕. */
    int read_ignore_subtitle;

    /**读,是否禁用延迟刷新.*/
    int read_nodelay;

    /**读,速率比例(3位小数). <=0 无效.*/
    int read_rate_scale;

    /**读,是否启用MP4流转换.*/
    int read_mp4toannexb;

    /**写,是否禁用延迟刷新. */
    int write_nodelay;

    /**写,启用FMP4封装. */
    int write_fmp4;

} abcdk_ffmpeg_editor_param_t;

typedef struct _abcdk_ffmpeg_editor abcdk_ffmpeg_editor_t;

/**释放.*/
void abcdk_ffmpeg_editor_free(abcdk_ffmpeg_editor_t **ctx);

/**创建.*/
abcdk_ffmpeg_editor_t *abcdk_ffmpeg_editor_alloc(int writer);

/**获取AVStream对象数量. */
int abcdk_ffmpeg_editor_stream_nb(abcdk_ffmpeg_editor_t *ctx);

/**获取AVStream对象指针.*/
AVStream *abcdk_ffmpeg_editor_stream_ctx(abcdk_ffmpeg_editor_t *ctx, int stream);

/**DTS/PTS转时间(秒).*/
double abcdk_ffmpeg_editor_stream_ts2sec(abcdk_ffmpeg_editor_t *ctx, int stream, int64_t ts);

/** DTS/PTS转序号.*/
int64_t abcdk_ffmpeg_editor_stream_ts2num(abcdk_ffmpeg_editor_t *ctx, int stream, int64_t ts);

/**打开. */
int abcdk_ffmpeg_editor_open(abcdk_ffmpeg_editor_t *ctx, const abcdk_ffmpeg_editor_param_t *param);

/**
 * 读数据包.
 * 
 * @return 0 成功, < 0 失败(出错或结束).
*/
int abcdk_ffmpeg_editor_read_packet(abcdk_ffmpeg_editor_t *ctx, AVPacket *dst);

/**
 * 创建流.
 * 
 * @return >= 0 成功(流索引), < 0 失败.
*/
int abcdk_ffmpeg_editor_add_stream(abcdk_ffmpeg_editor_t *ctx, const AVCodecContext *opt);

/**
 * 创建流.
 * 
 * @return >= 0 成功(流索引), < 0 失败.
*/
int abcdk_ffmpeg_editor_add_stream2(abcdk_ffmpeg_editor_t *ctx, const AVCodecParameters *opt, const AVRational *time_base,
                                    const AVRational *avg_frame_rate, const AVRational *r_frame_rate);

/**
 * 写数据包.
 * 
 * @return 0 成功, < 0 失败(出错或结束).
*/
int abcdk_ffmpeg_editor_write_packet(abcdk_ffmpeg_editor_t *ctx, AVPacket *src);

/**
 * 写头部信息.
 * 
 * @note 第一个数据包写入前会自动检测并执行, 如果不关注返回结果可以省略主动执行.
 * 
 * @return 0 成功, < 0 失败(出错或空间不足).
*/
int abcdk_ffmpeg_editor_write_header(abcdk_ffmpeg_editor_t *ctx);

/**
 * 写结尾信息.
 * 
 * @note 关闭时会自动检测并执行, 如果不关注返回结果可以省略主动执行.
 * 
 * @return 0 成功, < 0 失败(出错或空间不足).
*/
int abcdk_ffmpeg_editor_write_trailer(abcdk_ffmpeg_editor_t *ctx);


__END_DECLS

#endif // ABCDK_FFMPEG_EDITOR_H