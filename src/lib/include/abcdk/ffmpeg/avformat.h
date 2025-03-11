/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_AVFORMAT_H
#define ABCDK_FFMPEG_AVFORMAT_H

#include "abcdk/util/general.h"
#include "abcdk/util/string.h"
#include "abcdk/ffmpeg/avutil.h"
#include "abcdk/ffmpeg/avcodec.h"

__BEGIN_DECLS

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

/**
 * 释放自定义IO环境。
 */
void abcdk_avio_free(AVIOContext **ctx);

/**
 * 创建自定义IO环境。
 *
 * @param buffer_blocks 4K(字节)的倍数，默认值为8。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVIOContext *abcdk_avio_alloc(int buf_blocks, int write_flag, void *opaque);

/**
 * 打印流信息。
 */
void abcdk_avformat_dump(AVFormatContext *ctx,int is_output);

/**
 * 打印可选项。
 */
void abcdk_avformat_show_options(AVFormatContext *ctx);

/**
 * 释放AVFormatContext环境。
 *
 * @note 释放所有需要释放的内存和句柄。
 */
void abcdk_avformat_free(AVFormatContext **ctx);

/**
 * 创建流(输入)环境。
 *
 * @param interrupt 中断回调环境指针。
 * @param io 自定义IO环境指针。
 * @param dict 字典指针。!NULL(0) 需要调用者释放。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVFormatContext *abcdk_avformat_input_open(const char *short_name, const char *filename,
                                           AVIOInterruptCB *interrupt, AVIOContext *io,
                                           AVDictionary **dict);

/**
 * 探查流(输入)信息。
 *
 * @param dict 字典指针数组，数组的高度大于或等于ctx->nb_streams。!NULL(0) 需要调用者释放。
 *
 * @return >=0 成功，-1 失败。
 *
 */
int abcdk_avformat_input_probe(AVFormatContext *ctx, AVDictionary **dict);

/**
 * 读取流(输入)的数据包。
 *
 * @param only_type AVMEDIA_TYPE_NB 任意类型，!AVMEDIA_TYPE_NB 指定类型。
 *
 * @return >=0 成功，-1 失败。
 */
int abcdk_avformat_input_read(AVFormatContext *ctx, AVPacket *pkt, enum AVMediaType only_type);

/**
 * 过滤流(输入)的数据包。
 *
 * @param filter 过滤器指针。!NULL(0) 需要调用者释放。
 *
 * @return >=0 成功，-1 失败。
 *
 */
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58, 20, 100)
int abcdk_avformat_input_filter(AVFormatContext *ctx, AVPacket *pkt, AVBSFContext **filter);
#else
int abcdk_avformat_input_filter(AVFormatContext *ctx, AVPacket *pkt, AVBitStreamFilterContext **filter);
#endif

/**
 * 创建流(输出)环境。
 *
 * @param mime_type 媒体类型，NULL(0) 自动确定。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVFormatContext *abcdk_avformat_output_open(const char *short_name, const char *filename, const char *mime_type,
                                            AVIOInterruptCB *interrupt, AVIOContext *io);

/**
 * 创建新的流(输出)环境。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVStream *abcdk_avformat_output_stream(AVFormatContext *ctx, const AVCodec *codec);

/**
 * 创建新的流(输出)环境。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVStream *abcdk_avformat_output_stream2(AVFormatContext *ctx, const char *name);

/**
 * 创建新的流(输出)环境。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVStream *abcdk_avformat_output_stream3(AVFormatContext *ctx, enum AVCodecID id);

/**
 * 向流(输出)写入头部信息。
 *
 * @param dict 字典指针。!NULL(0) 需要调用者释放。
 *
 * @return >=0 成功，-1 失败。
 *
 */
int abcdk_avformat_output_header(AVFormatContext *ctx, AVDictionary **dict);

/**
 * 向流(输出)写入数据包。
 *
 * @param [in] flush 是否立即刷新。0 否，!0 是。
 *
 * @return 0 成功，!0 失败。
 *
 */
int abcdk_avformat_output_write(AVFormatContext *ctx, AVPacket *pkt,int flush);


/**
 * 向流(输出)写入结束信息。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_avformat_output_trailer(AVFormatContext *ctx);

/**
 * 从编/解码器环境复制参数。
 *
 * @return 0 成功，!0 失败。
 */
int abcdk_avstream_parameters_from_context(AVStream *vs, const AVCodecContext *ctx);

/**
 * 向编/解码器环境复制参数。
 *
 * @return 0 成功，!0 失败。
 */
int abcdk_avstream_parameters_to_context(AVCodecContext *ctx, const AVStream *vs);


/**
 * 获取流的时长(秒)。
 *
 * @return 秒.毫秒。
 */
double abcdk_avstream_duration(AVFormatContext *ctx, AVStream *vs,double xspeed);

/**
 * 获取FPS。
 *
 * @return 秒.毫秒。
 */
double abcdk_avstream_fps(AVFormatContext *ctx, AVStream *vs,double xspeed);

/**
 * DTS或PTS转自然时间。
 *
 * @return 秒.毫秒。
 */
double abcdk_avstream_ts2sec(AVFormatContext *ctx, AVStream *vs, int64_t ts,double xspeed);

/**
 * DTS或PTS转序号。
 *
 * @return 整型。
 */
int64_t abcdk_avstream_ts2num(AVFormatContext *ctx, AVStream *vs, int64_t ts,double xspeed);

/**
 * 获取指定流图像的宽。
 * 
 * @return 像素。
*/
int abcdk_avstream_width(AVFormatContext *ctx, AVStream *vs);

/**
 * 获取指定流图像的高。
 * 
 * @return 像素。
*/
int abcdk_avstream_height(AVFormatContext *ctx, AVStream *vs);

/**
 * 查找指定类型的流。
*/
AVStream *abcdk_avstream_find(AVFormatContext *ctx,enum AVMediaType type);

#endif // AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

__END_DECLS

#endif // ABCDK_FFMPEG_AVFORMAT_H
