/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_RTSP_H
#define ABCDK_RTSP_RTSP_H

#include "abcdk/util/defs.h"
#include "abcdk/openssl/openssl.h"

#ifdef OPENSSL_VERSION_NUMBER
#ifdef HAVE_LIVE555
#ifdef __cplusplus
#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include <RTSPServer.hh>
#endif //__cplusplus
#endif // HAVE_LIVE555
#endif //OPENSSL_VERSION_NUMBER

__BEGIN_DECLS

/** */
#define ABCDK_RTSP_SERVER_REALM "ABCDK MediaServer"

/**RTSP授权管理常量。*/
typedef enum _abcdk_rtsp_auth_constant
{
    /**无。*/
    ABCDK_RTSP_AUTH_NONE = -1,
#define ABCDK_RTSP_AUTH_NONE ABCDK_RTSP_AUTH_NONE

    /**常态。*/
    ABCDK_RTSP_AUTH_NORMAL = 1,
#define ABCDK_RTSP_AUTH_NORMAL ABCDK_RTSP_AUTH_NORMAL

    /**基于时间的一次性密码算法Time-based One-Time Password）。*/
    ABCDK_RTSP_AUTH_TOTP_SHA1 = 10,
#define ABCDK_RTSP_AUTH_TOTP_SHA1 ABCDK_RTSP_AUTH_TOTP_SHA1
#define ABCDK_RTSP_AUTH_TOTP_SHA128 ABCDK_RTSP_AUTH_TOTP_SHA1

    /**基于时间的一次性密码算法Time-based One-Time Password）。*/
    ABCDK_RTSP_AUTH_TOTP_SHA256 = 11,
#define ABCDK_RTSP_AUTH_TOTP_SHA256 ABCDK_RTSP_AUTH_TOTP_SHA256

    /**基于时间的一次性密码算法Time-based One-Time Password）。*/
    ABCDK_RTSP_AUTH_TOTP_SHA512 = 12,
#define ABCDK_RTSP_AUTH_TOTP_SHA512 ABCDK_RTSP_AUTH_TOTP_SHA512

} abcdk_rtsp_auth_constant_t;

/**RTSP媒体编码常量。*/
typedef enum _abcdk_rtsp_codec_constant
{
    ABCDK_RTSP_CODEC_NONE = -1,
#define ABCDK_RTSP_CODEC_NONE ABCDK_RTSP_CODEC_NONE

    ABCDK_RTSP_CODEC_H264 = 1,
#define ABCDK_RTSP_CODEC_H264 ABCDK_RTSP_CODEC_H264

    ABCDK_RTSP_CODEC_HEVC,
#define ABCDK_RTSP_CODEC_HEVC ABCDK_RTSP_CODEC_HEVC
#define ABCDK_RTSP_CODEC_H265 ABCDK_RTSP_CODEC_HEVC

    ABCDK_RTSP_CODEC_VP8,
#define ABCDK_RTSP_CODEC_VP8 ABCDK_RTSP_CODEC_VP8

    ABCDK_RTSP_CODEC_VP9,
#define ABCDK_RTSP_CODEC_VP9 ABCDK_RTSP_CODEC_VP9

    ABCDK_RTSP_CODEC_AAC = 100,
#define ABCDK_RTSP_CODEC_AAC ABCDK_RTSP_CODEC_AAC

    /*extdata[0] = channels,extdata[1]=sample_rate.*/
    ABCDK_RTSP_CODEC_PCM_MULAW = 200,
#define ABCDK_RTSP_CODEC_PCM_MULAW ABCDK_RTSP_CODEC_PCM_MULAW
#define ABCDK_RTSP_CODEC_G711U ABCDK_RTSP_CODEC_PCM_MULAW

    /*extdata[0] = channels,extdata[1]=sample_rate.*/
    ABCDK_RTSP_CODEC_PCM_ALAW,
#define ABCDK_RTSP_CODEC_PCM_ALAW ABCDK_RTSP_CODEC_PCM_ALAW
#define ABCDK_RTSP_CODEC_G711A ABCDK_RTSP_CODEC_PCM_ALAW

} abcdk_rtsp_codec_constant_t;


__END_DECLS

#endif //ABCDK_RTSP_RTSP_H