/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_AVCODEC_H
#define ABCDK_UTIL_AVCODEC_H

#include "abcdk/util/general.h"
#include "abcdk/util/avutil.h"
#include "abcdk/util/avformat.h"

__BEGIN_DECLS

#ifdef HAVE_FFMPEG

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif //__STDC_CONSTANT_MACROS

#include <libavcodec/avcodec.h>

#endif // HAVE_FFMPEG

#ifdef AVCODEC_AVCODEC_H

/**
 * 根据名字查找编/解码器。
 *
 * @param encode !0 查找编码器，0 查找解码器。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVCodec *abcdk_avcodec_find(const char *name, int encode);

/**
 * 根据ID查找编/解码器。
 *
 * @note h264、h265会优先尝试硬件加速。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVCodec *abcdk_avcodec_find2(enum AVCodecID id, int encode);

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
AVCodecContext *abcdk_avcodec_alloc2(const char *name, int encode);

/**
 * 创建编/解码器环境。
 *
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
 */
AVCodecContext *abcdk_avcodec_alloc3(enum AVCodecID id, int encode);

/**
 * 打开编/解码器环境。
 *
 * @param dict 字典，!NULL(0) 需要调用者释放。
 *
 * @return  >=0 成功，1 失败。
 */
int abcdk_avcodec_open(AVCodecContext *ctx, AVDictionary **dict);

/**
 * 解码。
 * 
 * @param in 数据包，NULL(0) 忽略输入，仅获取已解码的帧图。
 *
 * @return > 0 成功(解码帧数量)，0 延时解码，-1 失败，-2，未支持。
 *
 */
int abcdk_avcodec_decode(AVCodecContext *ctx, AVFrame *out, const AVPacket *in);

/**
 * 编码。
 *
 * @param in 数据包，NULL(0) 忽略输入，仅获取延时编码的数据包。
 *
 * @return > 0 成功(编码帧数量)，0 延时编码，-1 失败，-2，未支持。
 */
int abcdk_avcodec_encode(AVCodecContext *ctx, AVPacket *out, const AVFrame *in);

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
 */
void abcdk_avcodec_video_encode_prepare(AVCodecContext *ctx, int fps, int width, int height, int gop_size, int oformat_flags);

#endif // AVCODEC_AVCODEC_H

__END_DECLS

#endif // ABCDK_UTIL_AVCODEC_H