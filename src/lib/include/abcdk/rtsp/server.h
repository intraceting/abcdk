/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_H
#define ABCDK_RTSP_SERVER_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/rtsp/rtsp.h"

__BEGIN_DECLS

/**简单的RTSP服务。*/
typedef struct _abcdk_rtsp_server abcdk_rtsp_server_t;

/**销毁。*/
void abcdk_rtsp_server_destroy(abcdk_rtsp_server_t **ctx);

/**创建。*/
abcdk_rtsp_server_t *abcdk_rtsp_server_create(uint16_t port, const char  *realm);

/**启用TLS。*/
int abcdk_rtsp_server_enable_tls(abcdk_rtsp_server_t *ctx,const char *cert,const char *key);

/**停止。*/
void abcdk_rtsp_server_stop(abcdk_rtsp_server_t *ctx);

/**启动。*/
int abcdk_rtsp_server_start(abcdk_rtsp_server_t *ctx);

/**删除账户。*/
void abcdk_rtsp_server_remove_user(abcdk_rtsp_server_t *ctx, const char *username);

/**添加账户。*/
void abcdk_rtsp_server_add_user(abcdk_rtsp_server_t *ctx,  const char *username, const char *password);

/**删除媒体。*/
void abcdk_rtsp_server_remove_media(abcdk_rtsp_server_t *ctx, int media);

/**播放媒体。*/
int abcdk_rtsp_server_play_media(abcdk_rtsp_server_t *ctx,  int media);

/**创建媒体。*/
int abcdk_rtsp_server_create_media(abcdk_rtsp_server_t *ctx, const char *name, const char *info, const char *desc);

/**媒体添加流。*/
int abcdk_rtsp_server_media_add_stream(abcdk_rtsp_server_t *ctx, int media, int codec, abcdk_object_t *extdata, int cache);

/**媒体流附加数据。*/
int abcdk_rtsp_server_media_append_stream(abcdk_rtsp_server_t *ctx, int media, int stream, const void *data, size_t size, int64_t dts, int64_t pts, int64_t dur);




__END_DECLS

#endif //ABCDK_RTSP_SERVER_H