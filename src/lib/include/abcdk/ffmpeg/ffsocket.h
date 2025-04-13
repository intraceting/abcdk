/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_FFMPEG_FFSOCKET_H
#define ABCDK_FFMPEG_FFSOCKET_H

#include "abcdk/util/socket.h"
#include "abcdk/ffmpeg/ffeditor.h"

/*简单的IO通讯。*/
typedef struct _abcdk_ffsocket abcdk_ffsocket_t;

/**销毁。 */
void abcdk_ffsocket_destroy(abcdk_ffsocket_t **ctx);

/**创建。*/
abcdk_ffsocket_t *abcdk_ffsocket_create(const char *addr, int timeout, const char *cert, const char *key, const char *capath);

/**
 * 读。
 *
 * @return > 0 成功(长度)，0 断开或末尾，< 0 失败(参考AVERROR(n))
 */
int abcdk_ffsocket_read(void *opaque, uint8_t *buf, int size);

/**
 * 读。
 *
 * @return > 0 成功(长度)，0 断开或已满，< 0 失败(参考AVERROR(n))
 */
int abcdk_ffsocket_write(void *opaque, uint8_t *buf, int size);

#endif //ABCDK_FFMPEG_FFSOCKET_H