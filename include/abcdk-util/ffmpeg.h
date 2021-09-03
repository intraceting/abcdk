/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_FFMPEG_H
#define ABCDK_UTIL_FFMPEG_H

#include "abcdk-util/general.h"
#include "abcdk-util/image.h"

#ifdef HAVE_FFMPEG

__BEGIN_DECLS

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif //__STDC_CONSTANT_MACROS

#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/dict.h>
#include <libavutil/avutil.h>
#include <libavutil/base64.h>
#include <libavutil/common.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>

__END_DECLS

#endif //HAVE_FFMPEG

__BEGIN_DECLS

#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H) && defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)


/*------------------------------------------------------------------------------------------------*/

/**
 * 日志重定向到syslog。
*/
void abcdk_av_log2syslog();

/*------------------------------------------------------------------------------------------------*/

/**
 * 获取像素位宽。
 * 
 * @param padded 0 实际位宽，!0 存储位宽。
 * 
 * @return > 0 成功(像素位宽)，<= 0 失败。
*/
int abcdk_av_image_pixfmt_bits(enum AVPixelFormat pixfmt,int padded);

/**
 * 获取像素格式名字。
*/
const char* abcdk_av_image_pixfmt_name(enum AVPixelFormat pixfmt);

/**
 * 计算图像每个图层的高度。
 * 
 * @param pixfmt 像素格式
 * @param height 高(像素)
 * 
 * @return > 0 成功(图层数量)， <= 0 失败。
*/
int abcdk_av_image_fill_heights(int heights[4],int height,enum AVPixelFormat pixfmt);

/**
 * 计算图像每个图层的宽步长(字节)。
 * 
 * @param width 宽(像素)
 * @param align 对齐(字节)
 * 
 * @return > 0 成功(图层数量)， <= 0 失败。
*/
int abcdk_av_image_fill_strides(int strides[4],int width,int height,enum AVPixelFormat pixfmt,int align);

/**
 * 计算图像每个图层的宽步长(字节)。
 * 
 * @return > 0 成功(图层数量)， <= 0 失败。
*/
int abcdk_av_image_fill_strides2(abcdk_image_t *img,int align);

/**
 * 分派存储空间。
 * 
 * @param buffer 内存指针，传入NULL(0)。
 * 
 * @return >0 成功(分派的内存大小)， <= 0 失败。
*/
int abcdk_av_image_fill_pointers(uint8_t *datas[4],const int strides[4],int height,enum AVPixelFormat pixfmt,void *buffer);

/** 
 * 分派存储空间。
 * 
 * @return >0 成功(分派的内存大小)， <= 0 失败。
*/
int abcdk_av_image_fill_pointers2(abcdk_image_t *img,void *buffer);

/**
 * 计算需要的内存大小。
 * 
 * @return >0 成功(需要的内存大小)， <= 0 失败。
*/
int abcdk_av_image_size(const int strides[4],int height,enum AVPixelFormat pixfmt);

/**
 * 计算需要的内存大小。
 * 
 * @return >0 成功(需要的内存大小)， <= 0 失败。
*/
int abcdk_av_image_size2(int width,int height,enum AVPixelFormat pixfmt,int align);

/**
 * 计算需要的内存大小。
 * 
 * @return >0 成功(需要的内存大小)， <= 0 失败。
*/
int abcdk_av_image_size3(const abcdk_image_t *img);

/**
 * 图像复制。
 * 
*/
void abcdk_av_image_copy(uint8_t *dst_datas[4], int dst_strides[4], const uint8_t *src_datas[4], const int src_strides[4],
                         int width, int height, enum AVPixelFormat pixfmt);

/**
 * 图像复制。
 * 
*/
void abcdk_av_image_copy2(abcdk_image_t *dst, const abcdk_image_t *src);

/*------------------------------------------------------------------------------------------------*/

/**
 * 释放图像转换环境。
*/
void abcdk_sws_free(struct SwsContext **ctx);

/**
 * 创建图像转换环境。
 * 
 * @param flags 标志。SWS* 宏定义在swscale.h文件中。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
struct SwsContext *abcdk_sws_alloc(int src_width, int src_height, enum AVPixelFormat src_pixfmt,
                                   int dst_width, int dst_height, enum AVPixelFormat dst_pixfmt,
                                   int flags);

/**
 * 创建图像转换环境。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
struct SwsContext *abcdk_sws_alloc2(const abcdk_image_t *src, const abcdk_image_t *dst, int flags);

/*------------------------------------------------------------------------------------------------*/

/** 
 * 根据名字查找编/解码器。
 * 
 * @param encode !0 查找编码器，0 查找解码器。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
AVCodec *abcdk_avcodec_find(const char *name,int encode);

/**
 * 根据ID查找编/解码器。
 * 
 * @note h264、h265会优先尝试硬件加速。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
AVCodec *abcdk_avcodec_find2(enum AVCodecID id,int encode);

/**
 * 打印编/解码器可选项。
*/
void abcdk_avcodec_show_options(AVCodec *ctx);

/**
 * 释放编/解码器环境。
 * 
 */
void abcdk_avcodec_free(AVCodecContext **ctx);

/** 
 * 创建编/解码器环境。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
AVCodecContext *abcdk_avcodec_alloc(const AVCodec *ctx);

/** 
 * 创建编/解码器环境。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
AVCodecContext *abcdk_avcodec_alloc2(const char *name,int encode);

/** 
 * 创建编/解码器环境。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
AVCodecContext *abcdk_avcodec_alloc3(enum AVCodecID id,int encode);

/**
 * 打开编/解码器环境。
 * 
 * @param dict 字典，!NULL(0) 需要调用者释放。
 *  
 * @return  >=0 成功，1 失败。
*/
int abcdk_avcodec_open(AVCodecContext *ctx,AVDictionary **dict);

/**
 * 解码。
 * 
 * @return > 0 成功(解码帧数量)，0 延时解码，-1 失败，-2，未支持。
 * 
*/
int abcdk_avcodec_decode(AVCodecContext *ctx,AVFrame *out,const AVPacket *in);

/**
 * 编码。
 * 
 * @param in 数据包，NULL(0) 处理延时编码。
 * 
 * @return > 0 成功(编码帧数量)，0 延时编码，-1 失败，-2，未支持。
*/
int abcdk_avcodec_encode(AVCodecContext *ctx, AVPacket *out,const AVFrame *in);

/**
 * 配置视频编码环境基本参数。
 * 
 * @param fps 帧速。
 * @param width 宽(像素)。
 * @param height 高(像素)。
 * @param gop_size 关健帧间隔帧数，<= 0 使用帧速。
 * @param oformat_flags 输出的流标志。
 * 
 * @note 在abcdk_avcodec_open之前使用有效。
 * 
*/
void abcdk_avcodec_video_encode_prepare(AVCodecContext *ctx,int fps,int width,int height,int gop_size,int oformat_flags);


/*------------------------------------------------------------------------------------------------*/

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
AVIOContext *abcdk_avio_alloc(int buf_blocks,int write_flag,void *opaque);

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
 * @param interrupt_cb 中断回调环境指针。
 * @param io_cb 自定义IO环境指针。
 * @param dict 字典指针。!NULL(0) 需要调用者释放。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
AVFormatContext *abcdk_avformat_input_open(const char *short_name, const char *filename,
                                           AVIOInterruptCB *interrupt_cb, AVIOContext *io_cb,
                                           AVDictionary **dict);

/**
 * 探查流(输入)信息。
 * 
 * @param dict 字典指针数组，数组的高度大于或等于ctx->nb_streams。!NULL(0) 需要调用者释放。
 * @param dump !0 打印流信息，0 忽略。
 * 
 * @return >=0 成功，-1 失败。
 * 
*/
int abcdk_avformat_input_probe(AVFormatContext *ctx, AVDictionary **dict, int dump);

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
int abcdk_avformat_input_filter(AVFormatContext *ctx, AVPacket *pkt, AVBitStreamFilterContext **filter);

/**
 * 创建流(输出)环境。
 * 
 * @param mime_type 媒体类型，NULL(0) 自动确定。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
AVFormatContext *abcdk_avformat_output_open(const char *short_name, const char *filename, const char *mime_type,
                                            AVIOInterruptCB *interrupt_cb, AVIOContext *io_cb,
                                            AVDictionary **dict);

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
 * @param dump !0 打印流信息，0 忽略。
 * 
 * @return >=0 成功，-1 失败。
 * 
*/
int abcdk_avformat_output_header(AVFormatContext *ctx,AVDictionary **dict,int dump);

/**
 * 向流(输出)写入数据包。
 * 
 * @param vs 数据流，NULL(0) 写入结束包。
 * @param pkt 包，NULL(0) 写入结束包。
 * 
 * @return 0 成功，!0 失败。
 * 
*/
int abcdk_avformat_output_write(AVFormatContext *ctx, AVStream *vs, AVPacket *pkt);

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
 * @return 秒.毫秒
*/
double abcdk_avstream_get_duration(AVFormatContext *ctx,AVStream *vs);

/**
 * 获取FPS。
 * 
 * @return 秒.毫秒
*/
double abcdk_avstream_get_fps(AVFormatContext *ctx,AVStream *vs);

/**
 * DTS或PTS转自然时间。
 * 
 * @return 秒.毫秒
*/
double abcdk_avstream_ts2sec(AVFormatContext *ctx,AVStream *vs,int64_t ts);

/**
 * DTS或PTS转序号。
 * 
 * @return 整型。
*/
int64_t abcdk_avstream_ts2num(AVFormatContext *ctx, AVStream *vs,int64_t ts);

/*------------------------------------------------------------------------------------------------*/

#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H && AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H



__END_DECLS

#endif //ABCDK_UTIL_FFMPEG_H
