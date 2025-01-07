/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_FFMPEG_SWSCALE_H
#define ABCDK_FFMPEG_SWSCALE_H

#include "abcdk/util/general.h"
#include "abcdk/ffmpeg/avutil.h"

__BEGIN_DECLS

#ifdef HAVE_FFMPEG

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif //__STDC_CONSTANT_MACROS

#include <libswscale/swscale.h>

#endif //HAVE_FFMPEG

#ifdef SWSCALE_SWSCALE_H

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
struct SwsContext *abcdk_sws_alloc2(const AVFrame *src, const AVFrame *dst, int flags);

/** 
 * 转换图像格式。
 * 
 * @return > 0 成功(高度)，<= 0 失败。
*/
int abcdk_sws_scale(struct SwsContext *ctx,const AVFrame *src, AVFrame *dst);

#endif //SWSCALE_SWSCALE_H


__END_DECLS

#endif //ABCDK_FFMPEG_SWSCALE_H

