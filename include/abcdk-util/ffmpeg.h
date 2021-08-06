/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_FFMPEG_H
#define ABCDK_UTIL_FFMPEG_H

#include "abcdk-util/general.h"

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

#include <libswscale/swscale.h>

__END_DECLS

#endif //HAVE_FFMPEG

__BEGIN_DECLS

#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H)

/**
 * 简单的图像结构。
*/
typedef struct _abcdk_av_image
{
    /** 像素格式*/
    enum AVPixelFormat pixfmt;

    /** 数据指针*/
    uint8_t *datas[4];

    /** 宽步长(字节)*/
    int strides[4];

    /** 宽(像素)*/
    int width;

    /** 高(像素)*/
    int height;

}abcdk_av_image_t;


/** 检查像素格式是否支持。*/
#define ABCDK_AVPIXFMT_CHECK(pixfmt)   ((pixfmt) > AV_PIX_FMT_NONE && (pixfmt) < AV_PIX_FMT_NB)


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
int abcdk_av_image_pixfmt_bits(enum AVPixelFormat  pixfmt,int padded);

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
int abcdk_av_image_fill_strides2(abcdk_av_image_t *img,int align);

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
int abcdk_av_image_fill_pointers2(abcdk_av_image_t *img,void *buffer);

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
int abcdk_av_image_size3(const abcdk_av_image_t *img);

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
void abcdk_av_image_copy2(abcdk_av_image_t *dst, const abcdk_av_image_t *src);

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
struct SwsContext *abcdk_sws_alloc2(const abcdk_av_image_t *src, const abcdk_av_image_t *dst, int flags);

#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H

/*------------------------------------------------------------------------------------------------*/

__END_DECLS

#endif //ABCDK_UTIL_FFMPEG_H
