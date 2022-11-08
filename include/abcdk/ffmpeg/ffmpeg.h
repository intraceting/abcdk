/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_FFMPEG_FFMPEG_H
#define ABCDK_FFMPEG_FFMPEG_H

#include "abcdk/util/general.h"
#include "abcdk/util/time.h"
#include "abcdk/ffmpeg/avformat.h"
#include "abcdk/ffmpeg/avcodec.h"

__BEGIN_DECLS

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

/** FFMPEG对象。*/
typedef struct _abcdk_ffmpeg abcdk_ffmpeg_t;

/**
 * 销毁对象。
*/
void abcdk_ffmpeg_destroy(abcdk_ffmpeg_t **ctx);

/**
 * AVFormatContext对象指针。 
*/
AVFormatContext *abcdk_ffmpeg_ctxptr(abcdk_ffmpeg_t *ctx);

/**
 * 创建读者对象。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_ffmpeg_t *abcdk_ffmpeg_open_capture(const char *short_name, const char *url, const AVDictionary *dict);

/**
 * 读取数据包。
 * 
 * @param stream >=0 数据流索引，< 0 任意数据流。
 * 
 * @return >= 0 成功(数据流索引)，< 0 失败。
*/
int abcdk_ffmpeg_read(abcdk_ffmpeg_t *ctx, AVPacket *packet, int stream);

/**
 * 读取数据帧。
 * 
 * @param stream >=0 数据流索引，< 0 任意数据流。
 * 
 * @return >= 0 成功(数据流索引)，< 0 失败。
*/
int abcdk_ffmpeg_read2(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream);

/** 
 * 创建作者对象。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_ffmpeg_t *abcdk_ffmpeg_open_writer(const char*short_name,const char *url,const char *mime_type);

/**
 * 创建数据流。
 * 
 * @warning 仅支持视频流。
 * 
 * @param extdata 扩展数据的指针，NULL(0) 忽略。
 * @param extsize 扩展数据的长度。
 * @param have_codec 0 使用内部编码器(不支持B帧，并忽略外部扩展数据)，!0 使用外部编码器。
 * 
 * @return >= 0 成功(数据流索引)，< 0 失败。
*/
int abcdk_ffmpeg_add_stream(abcdk_ffmpeg_t *ctx, int fps, int width, int height, enum AVCodecID id,
                            const void *extdata, int extsize, int have_codec);

/**
 * 写入头部信息。
 * 
 * @param fmp4 0 默认的格式，!0 FMP4格式。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write_header(abcdk_ffmpeg_t *ctx,int fmp4);

/**
 * 写入结束信息。
 * 
 * @note 自动写入所有延时编码数据包。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write_trailer(abcdk_ffmpeg_t *ctx);

/**
 * 写入数据包(已编码)。
 * 
 * @note 如果是向网络流写入数据包，需要调用者限速调用此接口。因为过快的写入速度可能会造成网络数据包积压，连接被中断。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write(abcdk_ffmpeg_t *ctx, AVPacket *packet);

/**
 * 写入数据包(已编码)。
 * 
 * @warning 不支持B帧。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write2(abcdk_ffmpeg_t *ctx, void *data, int size, int stream);

/**
 * 写入数据帧(未编码)。
 * 
 * @warning 仅支持图像。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write3(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream);


#endif //AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H


__END_DECLS

#endif //ABCDK_FFMPEG_FFMPEG_H