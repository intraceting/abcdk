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
#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"

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
 * 查找指定类型的流。
*/
AVStream *abcdk_ffmpeg_find_stream(abcdk_ffmpeg_t *ctx,enum AVMediaType type);

/**
 * AVStream对象数量。
 */
int abcdk_ffmpeg_streams(abcdk_ffmpeg_t *ctx);

/**
 * AVStream对象指针。
*/
AVStream *abcdk_ffmpeg_streamptr(abcdk_ffmpeg_t *ctx,int stream);

/**
 * 获取流的时长(秒)。
 *
 * @return 秒.毫秒。
 */
double abcdk_ffmpeg_duration(abcdk_ffmpeg_t *ctx,int stream,double xspeed);

/**
 * 获取FPS。
 *
 * @return 秒.毫秒。
 */
double abcdk_ffmpeg_fps(abcdk_ffmpeg_t *ctx,int stream,double xspeed);

/**
 * DTS或PTS转自然时间。
 *
 * @return 秒.毫秒。
 */
double abcdk_ffmpeg_ts2sec(abcdk_ffmpeg_t *ctx,int stream, int64_t ts,double xspeed);

/**
 * DTS或PTS转序号。
 *
 * @return 整型。
 */
int64_t abcdk_ffmpeg_ts2num(abcdk_ffmpeg_t *ctx,int stream, int64_t ts,double xspeed);

/**
 * 获取指定流图像的宽。
 * 
 * @return 像素。
*/
int abcdk_ffmpeg_width(abcdk_ffmpeg_t *ctx,int stream);

/**
 * 获取指定流图像的高。
 * 
 * @return 像素。
*/
int abcdk_ffmpeg_height(abcdk_ffmpeg_t *ctx,int stream);

/**
 * 创建FFMPEG对象。
 * 
 * @param [in] opt 选项。
 * 
 * @code
 * --timeout < seconds >
 *  超时(秒)。
 * --mime-type < type >
 *  媒体类型。
 * --try-nvcodec < 1 | 0 >
 *  尝试NVCODEC编解码器。默认：0
 * --bit-stream-filter < 1 |0 >
 *  比特流过滤器。默认：0
 * @endcond
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_ffmpeg_t *abcdk_ffmpeg_open(int writer, const char *short_name, const char *url, AVIOContext *io, abcdk_option_t *opt);

/**
 * 创建读者对象。
 * 
 * @param [in] timeout 超时(秒)。
 *
*/ 
abcdk_ffmpeg_t *abcdk_ffmpeg_open_capture(const char *short_name, const char *url, int bsf, int timeout);

/** 
 * 创建作者对象。
 * 
 * @param [in] timeout 超时(秒)。注：仅对初步建立连接有效。
 * 
*/
abcdk_ffmpeg_t *abcdk_ffmpeg_open_writer(const char*short_name, const char *url, const char *mime_type, int timeout);

/**
 * 读延时。
 * 
 * @param [in] xspeed 倍速。
 * @param [in] stream >=0 流索引，< 0 使用最慢的流索引。
 * 
 */
void abcdk_ffmpeg_read_delay(abcdk_ffmpeg_t *ctx, double xspeed, int stream);

/**
 * 读取数据包。
 * 
 * @param stream >=0 流索引，< 0 任意流索引。
 * 
 * @return >= 0 成功(流索引)，< 0 失败(或结束)。
*/
int abcdk_ffmpeg_read(abcdk_ffmpeg_t *ctx, AVPacket *pkt, int stream);

/**
 * 读取数据帧。
 * 
 * @param stream >=0 流索引，< 0 任意流。
 * 
 * @return >= 0 成功(流索引)，< 0 失败(或结束)。
*/
int abcdk_ffmpeg_read2(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream);


/**
 * 创建流。
 * 
 * @param have_codec 0 使用内部编码器(不支持B帧，并忽略扩展数据)，!0 使用外部编码器。
 * 
 * @return >= 0 成功(流索引)，< 0 失败。
*/
int abcdk_ffmpeg_add_stream(abcdk_ffmpeg_t *ctx, const AVCodecContext *opt, int have_codec);

/**
 * 写入头部信息。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write_header0(abcdk_ffmpeg_t *ctx,const AVDictionary *dict);

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
int abcdk_ffmpeg_write(abcdk_ffmpeg_t *ctx, AVPacket *pkt, AVRational *src_time_base);

/**
 * 写入数据包(已编码)。
 * 
 * @note 不支持B帧。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write2(abcdk_ffmpeg_t *ctx, void *data, int size, int keyframe, int stream);

/**
 * 写入数据帧(未编码)。
 * 
 * @note 仅支持图像。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write3(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream);


#endif //AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H


__END_DECLS

#endif //ABCDK_FFMPEG_FFMPEG_H