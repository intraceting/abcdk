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

    /**超时(秒). */
    int timeout;

    /**读,是否忽略视频. */
    int read_ignore_video;

    /**读,是否忽略音频. */
    int read_ignore_audio;

    /**读,是否忽略字幕. */
    int read_ignore_subtitle;

    /**读,倍速(3位小数). <=0 无效.*/
    int read_speed_scale;

    /**读,是否启用MP4流转换.*/
    int read_mp4toannexb;

    /**写,是否禁用延迟刷新. */
    int write_nodelay;

    /**虚拟IO环境. */
    struct
    {
        int (*read_cb)(void *opaque, uint8_t *buf, int size);
        int (*write_cb)(void *opaque, uint8_t *buf, int size);
        void *opaque;
    } vio;

} abcdk_ffmpeg_editor_param_t;

typedef struct _abcdk_ffmpeg_editor abcdk_ffmpeg_editor_t;

/**释放.*/
void abcdk_ffmpeg_editor_free(abcdk_ffmpeg_editor_t **ctx);

/**创建.*/
abcdk_ffmpeg_editor_t *abcdk_ffmpeg_editor_alloc(int writer);

/**打开. */
int abcdk_ffmpeg_editor_open(abcdk_ffmpeg_editor_t *ctx, abcdk_ffmpeg_editor_param_t *param);

/** DTS/PTS转时间(秒).*/
double abcdk_ffmpeg_editor_ts2sec(abcdk_ffmpeg_editor_t *ctx,int stream, int64_t ts);

/**
 * 读数据包。
 * 
 * @return 0 成功, < 0 失败(出错或结束)。
*/
int abcdk_ffmpeg_editor_read_packet(abcdk_ffmpeg_editor_t *ctx, AVPacket *pkt);

__END_DECLS

#endif // ABCDK_FFMPEG_EDITOR_H