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

    ABCDK_MEDIA_PIXFMT_YUV420P,
#define ABCDK_MEDIA_PIXFMT_YUV420P ABCDK_MEDIA_PIXFMT_YUV420P

    ABCDK_MEDIA_PIXFMT_YUVJ420P = ABCDK_MEDIA_PIXFMT_YUV420P, //Similar YUV420P.
#define ABCDK_MEDIA_PIXFMT_YUVJ420P ABCDK_MEDIA_PIXFMT_YUVJ420P

    ABCDK_MEDIA_PIXFMT_I420,
#define ABCDK_MEDIA_PIXFMT_I420 ABCDK_MEDIA_PIXFMT_I420

    ABCDK_MEDIA_PIXFMT_YV12,
#define ABCDK_MEDIA_PIXFMT_YV12 ABCDK_MEDIA_PIXFMT_YV12

    ABCDK_MEDIA_PIXFMT_NV12,
#define ABCDK_MEDIA_PIXFMT_NV12 ABCDK_MEDIA_PIXFMT_NV12

    ABCDK_MEDIA_PIXFMT_NV21,
#define ABCDK_MEDIA_PIXFMT_NV21 ABCDK_MEDIA_PIXFMT_NV21

    ABCDK_MEDIA_PIXFMT_YUV422P,
#define ABCDK_MEDIA_PIXFMT_YUV422P ABCDK_MEDIA_PIXFMT_YUV422P

    ABCDK_MEDIA_PIXFMT_YUVJ422P = ABCDK_MEDIA_PIXFMT_YUV422P, //Similar YUV422P.
#define ABCDK_MEDIA_PIXFMT_YUVJ422P ABCDK_MEDIA_PIXFMT_YUVJ422P

    ABCDK_MEDIA_PIXFMT_YUYV,
#define ABCDK_MEDIA_PIXFMT_YUYV ABCDK_MEDIA_PIXFMT_YUYV

    ABCDK_MEDIA_PIXFMT_UYVY,
#define ABCDK_MEDIA_PIXFMT_UYVY ABCDK_MEDIA_PIXFMT_UYVY

    ABCDK_MEDIA_PIXFMT_NV16,
#define ABCDK_MEDIA_PIXFMT_NV16 ABCDK_MEDIA_PIXFMT_NV16

    ABCDK_MEDIA_PIXFMT_YUV444P,
#define ABCDK_MEDIA_PIXFMT_YUV444P ABCDK_MEDIA_PIXFMT_YUV444P

    ABCDK_MEDIA_PIXFMT_YUVJ444P = ABCDK_MEDIA_PIXFMT_YUV444P, //Similar YUV444P.
#define ABCDK_MEDIA_PIXFMT_YUVJ444P ABCDK_MEDIA_PIXFMT_YUVJ444P

    ABCDK_MEDIA_PIXFMT_NV24,
#define ABCDK_MEDIA_PIXFMT_NV24 ABCDK_MEDIA_PIXFMT_NV24

    ABCDK_MEDIA_PIXFMT_YUV411P,
#define ABCDK_MEDIA_PIXFMT_YUV411P ABCDK_MEDIA_PIXFMT_YUV411P

    ABCDK_MEDIA_PIXFMT_RGB24,
#define ABCDK_MEDIA_PIXFMT_RGB24 ABCDK_MEDIA_PIXFMT_RGB24

    ABCDK_MEDIA_PIXFMT_BGR24,
#define ABCDK_MEDIA_PIXFMT_BGR24 ABCDK_MEDIA_PIXFMT_BGR24

    ABCDK_MEDIA_PIXFMT_RGB32,
#define ABCDK_MEDIA_PIXFMT_RGB32 ABCDK_MEDIA_PIXFMT_RGB32

    ABCDK_MEDIA_PIXFMT_BGR32,
#define ABCDK_MEDIA_PIXFMT_BGR32 ABCDK_MEDIA_PIXFMT_BGR32

    ABCDK_MEDIA_PIXFMT_GRAY8,
#define ABCDK_MEDIA_PIXFMT_GRAY8 ABCDK_MEDIA_PIXFMT_GRAY8

    ABCDK_MEDIA_PIXFMT_GRAYF32,
#define ABCDK_MEDIA_PIXFMT_GRAYF32 ABCDK_MEDIA_PIXFMT_GRAYF32

} abcdk_media_pixfmt_constant_t;

/**获取通道数量。*/
int abcdk_media_pixfmt_channels(int pixfmt);

/**转为ffmpeg类型。*/
int abcdk_media_pixfmt_to_ffmpeg(int pixfmt);

/**从ffmpeg类型转。*/
int abcdk_media_pixfmt_from_ffmpeg(int pixfmt);

__END_DECLS

#endif // ABCDK_MEDIA_PIXFMT_H