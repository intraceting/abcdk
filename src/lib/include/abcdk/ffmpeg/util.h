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
void abcdk_ffmpeg_deinit();

/**初始化库环境. */
void abcdk_ffmpeg_init();

/**日志重定向. */
void abcdk_ffmpeg_log_redirect();

/**释放.*/
void abcdk_ffmpeg_io_free(AVIOContext **ctx);

/**创建.*/
AVIOContext *abcdk_ffmpeg_io_alloc(int buf_blocks, int write_flag);

/**打印媒体信息. */
void abcdk_ffmpeg_media_dump(AVFormatContext *ctx, int output);

/**打印媒体选项信息. */
void abcdk_ffmpeg_media_option_dump(AVFormatContext *ctx);

/**释放.*/
void abcdk_ffmpeg_media_free(AVFormatContext **ctx);

/**打印编码信息. */
void abcdk_ffmpeg_codec_option_dump(AVCodec *ctx);

/**释放. */
void abcdk_ffmpeg_codec_free(AVCodecContext **ctx);

/**
 * 转成浮点数.
 *
 * @param [in] scale 缩放比例.
 */
double abcdk_ffmpeg_q2d(AVRational *r, double scale);

/**时间基值转浮点.*/
double abcdk_ffmpeg_stream_timebase_q2d(AVStream *vs_ctx, double scale);

/**获取时长(秒).*/
double abcdk_ffmpeg_stream_duration(AVStream *vs_ctx, double scale);

/**获取时间速率.*/
double abcdk_ffmpeg_stream_time2rate(AVStream *vs_ctx, double scale);

/**DTS/PTS转时间(秒).*/
double abcdk_ffmpeg_stream_ts2sec(AVStream *vs_ctx, int64_t ts, double scale);

/**DTS/PTS转序号.*/
int64_t abcdk_ffmpeg_stream_ts2num(AVStream *vs_ctx, int64_t ts, double scale);

/**修复比特速率.*/
void abcdk_ffmpeg_stream_fix_bitrate(AVStream *vs_ctx);

/**获取像素位宽.*/
int abcdk_ffmpeg_pixfmt_get_bit(AVPixelFormat pixfmt, int have_pad);

/**获取像素格式名字. */
const char *abcdk_ffmpeg_pixfmt_get_name(AVPixelFormat pixfmt);

/**获取像素格式通道数. */
int abcdk_ffmpeg_pixfmt_get_channel(AVPixelFormat pixfmt);

/**
 * 计算图像每个图层的高度.
 *
 * @return > 0 图层数量, <= 0 失败(格式错误或其它).
 */
int abcdk_ffmpeg_image_fill_height(int heights[4], int height, AVPixelFormat pixfmt);

/**
 * 计算图像每个图层的宽步长(字节).
 *
 * @param align 行对齐(字节).
 *
 * @return > 0 图层数量, <= 0 失败(格式错误或其它).
 */
int abcdk_ffmpeg_image_fill_stride(int stride[4], int width, AVPixelFormat pixfmt, int align);

/**
 * 分派存储空间.
 *
 * @param buffer 内存指针. 允许为NULL(0).
 *
 * @return > 0 分派的内存大小,  <= 0 失败(格式错误或其它).
 */
int abcdk_ffmpeg_image_fill_pointer(uint8_t *data[4], const int stride[4], int height, AVPixelFormat pixfmt, void *buffer);

/**
 * 计算图像需要的内存大小.
 *
 * @return > 0 分派的内存大小,  <= 0 失败(格式错误或其它).
 */
int abcdk_ffmpeg_image_get_size(const int stride[4], int height, AVPixelFormat pixfmt);

/**
 * 计算图像需要的内存大小.
 *
 * @param align 行对齐(字节).
 */
int abcdk_ffmpeg_image_get_size2(int width, int height, AVPixelFormat pixfmt, int align);

/**图像复制.*/
void abcdk_ffmpeg_image_copy(uint8_t *dst_data[4], int dst_stride[4],
                             const uint8_t *src_data[4], const int src_stride[4],
                             int width, int height, AVPixelFormat pixfmt);

/**图像复制.*/
void abcdk_ffmpeg_image_copy2(AVFrame *dst, const AVFrame *src);

__END_DECLS

#endif // ABCDK_FFMPEG_UTIL_H