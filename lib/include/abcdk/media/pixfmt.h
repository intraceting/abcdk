/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_PIXFMT_H
#define ABCDK_MEDIA_PIXFMT_H

#include "abcdk/util/general.h"
#include "abcdk/ffmpeg/avutil.h"

__BEGIN_DECLS

/**常量。*/
typedef enum _abcdk_media_pixfmt_constant
{
    ABCDK_MEDIA_PIXFMT_NONE = -1,
#define ABCDK_MEDIA_PIXFMT_NONE ABCDK_MEDIA_PIXFMT_NONE

    ABCDK_MEDIA_PIXFMT_YUV410P = 1, ///< planar YUV 4:1:0,  9bpp, (1 Cr & Cb sample per 4x4 Y samples)
#define ABCDK_MEDIA_PIXFMT_YUV410P ABCDK_MEDIA_PIXFMT_YUV410P

    ABCDK_MEDIA_PIXFMT_YUV411P = 2, ///< planar YUV 4:1:1, 12bpp, (1 Cr & Cb sample per 4x1 Y samples)
#define ABCDK_MEDIA_PIXFMT_YUV411P ABCDK_MEDIA_PIXFMT_YUV411P

    ABCDK_MEDIA_PIXFMT_YUV420P = 3, ///< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
#define ABCDK_MEDIA_PIXFMT_YUV420P ABCDK_MEDIA_PIXFMT_YUV420P

    ABCDK_MEDIA_PIXFMT_YUV422P = 4, ///< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
#define ABCDK_MEDIA_PIXFMT_YUV422P ABCDK_MEDIA_PIXFMT_YUV422P

    ABCDK_MEDIA_PIXFMT_YUV444P = 5, ///< planar YUV 4:4:4, 24bpp, (1 Cr & Cb sample per 1x1 Y samples)
#define ABCDK_MEDIA_PIXFMT_YUV444P ABCDK_MEDIA_PIXFMT_YUV444P

    ABCDK_MEDIA_PIXFMT_YUVJ411P = 6,    ///< planar YUV 4:1:1, 12bpp, (1 Cr & Cb sample per 4x1 Y samples) full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV411P and setting color_range
#define ABCDK_MEDIA_PIXFMT_YUVJ411P ABCDK_MEDIA_PIXFMT_YUVJ411P

    ABCDK_MEDIA_PIXFMT_YUVJ420P = 7, ///< planar YUV 4:2:0, 12bpp, full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV420P and setting color_range
#define ABCDK_MEDIA_PIXFMT_YUVJ420P ABCDK_MEDIA_PIXFMT_YUVJ420P

    ABCDK_MEDIA_PIXFMT_YUVJ422P = 8, ///< planar YUV 4:2:2, 16bpp, full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV422P and setting color_range
#define ABCDK_MEDIA_PIXFMT_YUVJ422P ABCDK_MEDIA_PIXFMT_YUVJ422P

    ABCDK_MEDIA_PIXFMT_YUVJ444P = 9, ///< planar YUV 4:4:4, 24bpp, full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV444P and setting color_range
#define ABCDK_MEDIA_PIXFMT_YUVJ444P ABCDK_MEDIA_PIXFMT_YUVJ444P

    ABCDK_MEDIA_PIXFMT_YUYV422 = 10, ///< packed YUV 4:2:2, 16bpp, Y0 Cb Y1 Cr
#define ABCDK_MEDIA_PIXFMT_YUYV422 ABCDK_MEDIA_PIXFMT_YUYV422

    ABCDK_MEDIA_PIXFMT_UYVY422 = 11, ///< packed YUV 4:2:2, 16bpp, Cb Y0 Cr Y1
#define ABCDK_MEDIA_PIXFMT_UYVY422 ABCDK_MEDIA_PIXFMT_UYVY422

    ABCDK_MEDIA_PIXFMT_NV12 = 1000, ///< planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
#define ABCDK_MEDIA_PIXFMT_NV12 ABCDK_MEDIA_PIXFMT_NV12

    ABCDK_MEDIA_PIXFMT_NV21 = 1001, ///< as above, but U and V bytes are swapped
#define ABCDK_MEDIA_PIXFMT_NV21 ABCDK_MEDIA_PIXFMT_NV21

    ABCDK_MEDIA_PIXFMT_NV16 = 1002, ///< interleaved chroma YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
#define ABCDK_MEDIA_PIXFMT_NV16 ABCDK_MEDIA_PIXFMT_NV16

    ABCDK_MEDIA_PIXFMT_NV24 = 1003, ///< planar YUV 4:4:4, 24bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
#define ABCDK_MEDIA_PIXFMT_NV24 ABCDK_MEDIA_PIXFMT_NV24

    ABCDK_MEDIA_PIXFMT_NV42 = 1004, ///< as above, but U and V bytes are swapped
#define ABCDK_MEDIA_PIXFMT_NV42 ABCDK_MEDIA_PIXFMT_NV42

    ABCDK_MEDIA_PIXFMT_GRAY8 = 2000,
#define ABCDK_MEDIA_PIXFMT_GRAY8 ABCDK_MEDIA_PIXFMT_GRAY8

    ABCDK_MEDIA_PIXFMT_RGB24 = 2001, ///< packed RGB 8:8:8, 24bpp, RGBRGB...
#define ABCDK_MEDIA_PIXFMT_RGB24 ABCDK_MEDIA_PIXFMT_RGB24

    ABCDK_MEDIA_PIXFMT_BGR24 = 2002, ///< packed RGB 8:8:8, 24bpp, BGRBGR...
#define ABCDK_MEDIA_PIXFMT_BGR24 ABCDK_MEDIA_PIXFMT_BGR24

    ABCDK_MEDIA_PIXFMT_RGB32 = 2003,
#define ABCDK_MEDIA_PIXFMT_RGB32 ABCDK_MEDIA_PIXFMT_RGB32

    ABCDK_MEDIA_PIXFMT_BGR32 = 2004,
#define ABCDK_MEDIA_PIXFMT_BGR32 ABCDK_MEDIA_PIXFMT_BGR32

    ABCDK_MEDIA_PIXFMT_GRAYF32 = 2005,
#define ABCDK_MEDIA_PIXFMT_GRAYF32 ABCDK_MEDIA_PIXFMT_GRAYF32

} abcdk_media_pixfmt_constant_t;

/**获取通道数量。*/
int abcdk_media_pixfmt_channels(int format);

/**获取通道位宽。*/
int abcdk_media_pixfmt_bitwidth(int format, int plane);

/**转为ffmpeg类型。*/
int abcdk_media_pixfmt_to_ffmpeg(int format);

/**从ffmpeg类型转。*/
int abcdk_media_pixfmt_from_ffmpeg(int format);

__END_DECLS

#endif // ABCDK_MEDIA_PIXFMT_H