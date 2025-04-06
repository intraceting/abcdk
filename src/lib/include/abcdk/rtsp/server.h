/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_H
#define ABCDK_RTSP_SERVER_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

/**简单的RTSP服务。*/
typedef struct _abcdk_rtsp_server abcdk_rtsp_server_t;

/*销毁。*/
void abcdk_rtsp_server_destroy(abcdk_rtsp_server_t **ctx);

abcdk_rtsp_server_t *abcdk_rtsp_server_create(uint16_t port, char const *realm);

__END_DECLS

#endif //ABCDK_RTSP_SERVER_H