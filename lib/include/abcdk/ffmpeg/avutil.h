/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_AVUTIL_H
#define ABCDK_FFMPEG_AVUTIL_H

#include "abcdk/util/general.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS

#ifdef HAVE_FFMPEG

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
#include <libavutil/frame.h>
#include <libavutil/rational.h>

#endif // HAVE_FFMPEG

#ifdef AVUTIL_AVUTIL_H

/**
 * R2D(num/den)。
 * 
 * @param scale 比例。
 * 
*/
double abcdk_avmatch_r2d(AVRational r, double scale);

/**
 * 获取像素位宽。
 *
 * @param padded 0 实际位宽，!0 存储位宽。
 *
 * @return > 0 成功(像素位宽)，<= 0 失败。
 */
int abcdk_avimage_pixfmt_bits(enum AVPixelFormat pixfmt, int padded);

/**
 * 获取像素格式名字。
 */
const char *abcdk_avimage_pixfmt_name(enum AVPixelFormat pixfmt);

/**
 * 计算图像每个图层的高度。
 *
 * @param pixfmt 像素格式
 * @param height 高(像素)
 *
 * @return > 0 成功(图层数量)， <= 0 失败。
 */
int abcdk_avimage_fill_heights(int heights[4], int height, enum AVPixelFormat pixfmt);

/**
 * 计算图像每个图层的宽步长(字节)。
 *
 * @param width 宽(像素)
 * @param align 对齐(字节)
 *
 * @return > 0 成功(图层数量)， <= 0 失败。
 */
int abcdk_avimage_fill_strides(int strides[4], int width, int height, enum AVPixelFormat pixfmt, int align);

/**
 * 分派存储空间。
 *
 * @param buffer 内存指针，传入NULL(0)。
 *
 * @return >0 成功(分派的内存大小)， <= 0 失败。
 */
int abcdk_avimage_fill_pointers(uint8_t *datas[4], const int strides[4], int height,
                                enum AVPixelFormat pixfmt, void *buffer);

/**
 * 计算需要的内存大小。
 *
 * @return >0 成功(需要的内存大小)， <= 0 失败。
 */
int abcdk_avimage_size(const int strides[4], int height, enum AVPixelFormat pixfmt);

/**
 * 计算需要的内存大小。
 *
 * @return >0 成功(需要的内存大小)， <= 0 失败。
 */
int abcdk_avimage_size2(int width, int height, enum AVPixelFormat pixfmt, int align);

/**
 * 图像复制。
 */
void abcdk_avimage_copy(uint8_t *dst_datas[4], int dst_strides[4], const uint8_t *src_datas[4],
                        const int src_strides[4], int width, int height, enum AVPixelFormat pixfmt);

/**
 * 帧复制。
 */
void abcdk_avframe_copy(AVFrame *dst, const AVFrame *src);

/**重定向到轨迹日志。*/
void abcdk_avlog_redirect2trace();

#endif // AVUTIL_AVUTIL_H

__END_DECLS

#endif // ABCDK_FFMPEG_AVUTIL_H
