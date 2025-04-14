/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_RELAY_H
#define ABCDK_RTSP_RELAY_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/ffmpeg/ffeditor.h"
#include "abcdk/ffmpeg/ffsocket.h"
#include "abcdk/rtsp/server.h"

__BEGIN_DECLS

/**简单的RTSP中继服务。*/
typedef struct _abcdk_rtsp_relay abcdk_rtsp_relay_t;

/**销毁。*/
void abcdk_rtsp_relay_destroy(abcdk_rtsp_relay_t **ctx);

/**
 * 创建。
 *
 * @note rtmp[s]://PATHFILE
 * @note rtsp[s]://PATHFILE
 * @note http[s]://PATHFILE
 * @note /PATHFILE
 *
 * @param [in] src_timeout 超时(秒)。
 * @param [in] src_retry 重试间隔(秒)。
 */
abcdk_rtsp_relay_t *abcdk_rtsp_relay_create(abcdk_rtsp_server_t *server_ctx, const char *media_name, const char *src_url, const char *src_fmt, float src_xspeed, int src_timeout, int src_retry);

__END_DECLS

#endif //ABCDK_RTSP_RELAY_H
