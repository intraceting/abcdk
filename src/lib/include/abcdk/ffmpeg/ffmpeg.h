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

#include <libswscale/swscale.h>

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 91, 100)
#include <libavcodec/bsf.h>
#endif //#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(58, 91, 100)

//
typedef enum AVCodecID AVCodecID;
typedef enum AVPixelFormat AVPixelFormat;

__END_DECLS
#endif // HAVE_FFMPEG

#ifndef AVUTIL_FRAME_H
typedef struct AVFrame AVFrame;
#endif //#ifndef AVUTIL_FRAME_H

#ifndef AVCODEC_AVCODEC_H
typedef struct AVPacket AVPacket;
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

/** 最大支持16个。*/
#define ABCDK_FFMPEG_MAX_STREAMS 16

#endif //ABCDK_FFMPEG_FFMPEG_H
