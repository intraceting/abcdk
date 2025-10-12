/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_UTIL_H
#define ABCDK_FFMPEG_UTIL_H

#include "abcdk/util/trace.h"
#include "abcdk/util/string.h"
#include "abcdk/ffmpeg/ffmpeg.h"

__BEGIN_DECLS

/**反初始化库环境. */
void abcdk_ffmpeg_library_deinit();

/**初始化库环境. */
void abcdk_ffmpeg_library_init();

/**日志重定向. */
void abcdk_ffmpeg_log_redirect();


/**释放.*/
void abcdk_ffmpeg_io_free(AVIOContext **ctx);

/**创建.*/
AVIOContext *abcdk_ffmpeg_io_alloc(int buf_blocks, int write_flag);

/**打印媒体信息. */
void abcdk_ffmpeg_media_dump(AVFormatContext *ctx,int output);

/**打印媒体选项信息. */
void abcdk_ffmpeg_media_option_dump(AVFormatContext *ctx);

/**释放.*/
void abcdk_ffmpeg_media_free(AVFormatContext **ctx);

/**打开输入.*/
int abcdk_avformat_media_open_input(AVFormatContext **ctx, const char *fmt, const char *url, AVIOContext *vio_ctx,
                                    AVIOInterruptCB *inter_ctx, AVDictionary **options);

/**打开输出.*/
int abcdk_avformat_media_open_output(AVFormatContext **ctx, const char *fmt, const char *url, AVIOContext *vio_ctx,
                                    AVIOInterruptCB *inter_ctx, AVDictionary **options);

/**打印编码信息. */
void abcdk_ffmpeg_codec_option_dump(AVCodec *ctx);

/**释放. */
void abcdk_ffmpeg_codec_free(AVCodecContext **ctx);

/**
 * 转成浮点数.
 * 
 * @param [in] scale 缩放比例.
*/
double abcdk_ffmpeg_q2d(AVRational r, double scale);

/** 时间基值转浮点.*/
double abcdk_ffmpeg_stream_timebase_q2d(AVStream *vs_ctx,double scale);

/** 获取时长(秒).*/
double abcdk_ffmpeg_stream_duration(AVStream *vs_ctx, double scale);

/** 获取帧率.*/
double abcdk_ffmpeg_stream_fps(AVFormatContext *ctx, AVStream *vs_ctx,double scale);

/** DTS/PTS转时间(秒).*/
double abcdk_ffmpeg_stream_ts2sec(AVStream *vs_ctx, int64_t ts, double scale);


__END_DECLS

#endif // ABCDK_FFMPEG_UTIL_H