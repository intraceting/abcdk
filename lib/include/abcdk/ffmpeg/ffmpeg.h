/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
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

/** FFMPEG对象配置。*/
typedef struct _abcdk_ffmpeg_config
{
    /** 角色。0 读者，!0 作者。*/
    int writer;

    /** 
     * 自定义IO。
     * 
     * @note 当读写回调函数存在时有效。
    */
    struct
    {   
        /** 
         * 缓存大小。
         * 
         * @note 4KB的倍数。
        */
        int buffer_size;

        /** 应用环境指针。*/
        void *opaque;

        /** 读回调。 */
        int (*read_cb)(void *opaque, uint8_t *buf, int size);

        /** 写回调。 */
        int (*write_cb)(void *opaque, uint8_t *buf, int size);
    } io;

    /** 超时(秒)。*/
    int timeout;

    /** 
     * 文件名或资源名的简写名称。
     * 
     * @note 允许为NULL(0)。
    */
    const char *short_name;

    /** 文件名或资源名的完整名称。*/
    const char *file_name;

    /** 
     * MIME类型。
     * 
     * @note 作者有效，允许为NULL(0)。
    */
    const char *mime_type;

    /** 尝试NVCODEC编/解码器。0 否，!0 是。*/
    int try_nvcodec;

    /** 
     * 比特流过滤器。0 不启用，!0 启用。
     * 
     * @note MP4封装格式，H264或HEVC编码有效。
     */
    int bit_stream_filter;

    /** 
     * 读刷新。0 不启用，!0 启用。
     * 
     * @note 影响播放，需应用层自行控制播放速度。
    */
    float read_flush;

    /** 读的速度(倍数)。0.01~100.0 */
    float read_speed;

    /** 读的最大延迟(秒.毫秒)。0.020~86400.0*/
    float read_delay_max;

    /** 
     * 写刷新。0 不启用，!0 启用。
     * 
     * @note 影响效率，但推流会减少延时。
    */
    int write_flush;

}abcdk_ffmpeg_config_t;

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
double abcdk_ffmpeg_duration(abcdk_ffmpeg_t *ctx,int stream);

/**
 * 获取FPS。
 *
 * @return 秒.毫秒。
 */
double abcdk_ffmpeg_fps(abcdk_ffmpeg_t *ctx,int stream);

/**
 * DTS或PTS转自然时间。
 *
 * @return 秒.毫秒。
 */
double abcdk_ffmpeg_ts2sec(abcdk_ffmpeg_t *ctx,int stream, int64_t ts);

/**
 * DTS或PTS转序号。
 *
 * @return 整型。
 */
int64_t abcdk_ffmpeg_ts2num(abcdk_ffmpeg_t *ctx,int stream, int64_t ts);

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
 * @note 在对象关闭前，配置信息必须保持有效且不能更改。
 *  
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_ffmpeg_t *abcdk_ffmpeg_open(abcdk_ffmpeg_config_t *cfg);

/**
 * 读取数据包。
 * 
 * @return >= 0 成功(流索引)，< 0 失败(或结束)。
*/
int abcdk_ffmpeg_read_packet(abcdk_ffmpeg_t *ctx, AVPacket *pkt, int stream);

/**
 * 读取数据帧(已解码)。
 * 
 * @param stream 流索引。 >=0 有效，< 0 无效。
 * 
 * @return >= 0 成功(流索引)，< 0 失败(或结束)。
*/
int abcdk_ffmpeg_read_frame(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream);

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
int abcdk_ffmpeg_write_packet(abcdk_ffmpeg_t *ctx, AVPacket *pkt, AVRational *src_time_base);

/**
 * 写入数据包(已编码)。
 * 
 * @note 不支持B帧。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write_packet2(abcdk_ffmpeg_t *ctx, void *data, int size, int keyframe, int stream);

/**
 * 写入数据帧(未编码)。
 * 
 * @note 仅支持图像和音频。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_ffmpeg_write_frame(abcdk_ffmpeg_t *ctx, AVFrame *frame, int stream);


#endif //AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H


__END_DECLS

#endif //ABCDK_FFMPEG_FFMPEG_H