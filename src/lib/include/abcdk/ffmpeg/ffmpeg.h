/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_FFMPEG_FFMPEG_H
#define ABCDK_FFMPEG_FFMPEG_H

#include "abcdk/util/general.h"

#ifdef HAVE_FFMPEG
__BEGIN_DECLS

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif //__STDC_CONSTANT_MACROS

#include <libavcodec/version.h>
#include <libavcodec/avcodec.h>

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 91, 100)
#include <libavcodec/bsf.h>
#endif //#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 91, 100)

#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>

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
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>

#include <libswscale/swscale.h>

//Alias.
typedef enum AVCodecID AVCodecID;
typedef enum AVPixelFormat AVPixelFormat;
typedef enum AVHWDeviceType AVHWDeviceType;

__END_DECLS
#else //#ifdef HAVE_FFMPEG

#define AV_VERSION_INT(a, b, c) ((a)<<16 | (b)<<8 | (c))
#define AV_VERSION_DOT(a, b, c) a ##.## b ##.## c
#define AV_VERSION(a, b, c) AV_VERSION_DOT(a, b, c)

//Alias.
typedef int AVCodecID;
typedef int AVPixelFormat;
typedef int AVHWDeviceType;

#endif //#ifdef HAVE_FFMPEG


#ifndef AVUTIL_FRAME_H
typedef struct AVFrame AVFrame;
#endif //#ifndef AVUTIL_FRAME_H

#ifndef AVCODEC_AVCODEC_H
typedef struct AVPacket AVPacket;
typedef struct AVCodec AVCodec;
typedef struct AVCodecContext AVCodecContext;
typedef struct AVCodecParameters AVCodecParameters;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)
typedef struct AVBitStreamFilter AVBitStreamFilter;
typedef struct AVBSFContext AVBSFContext;
#endif //#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 91, 100)
#endif //#ifndef AVCODEC_AVCODEC_H

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 91, 100)
#ifndef AVCODEC_BSF_H
typedef struct AVBitStreamFilter AVBitStreamFilter;
typedef struct AVBSFContext AVBSFContext;
#endif //#ifndef AVCODEC_BSF_H
#endif //#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 91, 100)

#ifndef AVUTIL_RATIONAL_H
typedef struct AVRational AVRational;
#endif //#ifndef AVUTIL_RATIONAL_H

#ifndef AVFORMAT_AVIO_H
typedef struct AVIOContext AVIOContext;
typedef struct AVIOInterruptCB AVIOInterruptCB;
#endif //#ifndef AVFORMAT_AVIO_H

#ifndef AVFORMAT_AVFORMAT_H
typedef struct AVFormatContext AVFormatContext;
typedef struct AVStream AVStream;
#endif //#ifndef AVFORMAT_AVFORMAT_H

#ifndef AVUTIL_DICT_H
typedef struct AVDictionary AVDictionary;
#endif //#ifndef AVUTIL_DICT_H

/** 最大支持16个。*/
#define ABCDK_FFMPEG_MAX_STREAMS 16

#endif //ABCDK_FFMPEG_FFMPEG_H
