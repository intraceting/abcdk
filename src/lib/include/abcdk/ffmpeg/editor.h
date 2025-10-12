/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_EDITOR_H
#define ABCDK_FFMPEG_EDITOR_H

#include "abcdk/util/trace.h"
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/util.h"
#include "abcdk/util/time.h"

__BEGIN_DECLS

typedef struct _abcdk_editor_param
{
    /**格式.*/
    const char *fmt;

    /**地址. */
    const char *url;

    /**超时(秒). */
    int timeout;

    /**读倍速.*/
    float read_speed;

    /**读最大延迟(秒.毫秒).*/
    float read_max_delay;

    /**写禁用延迟. */
    int write_nodelay;

    /**虚拟IO. */
    struct
    {
        int (*read_cb)(void *opaque, uint8_t *buf, int size);
        int (*write_cb)(void *opaque, uint8_t *buf, int size);
        void *opaque;

    } vio;

} abcdk_editor_param_t;

typedef struct _abcdk_ffmpeg_editor abcdk_ffmpeg_editor_t;

void abcdk_ffmpeg_editor_free(abcdk_ffmpeg_editor_t **ctx);
abcdk_ffmpeg_editor_t *abcdk_ffmpeg_editor_alloc(int writer);

int abcdk_ffmpeg_editor_open(abcdk_ffmpeg_editor_t *ctx, abcdk_editor_param_t *param);

__END_DECLS

#endif // ABCDK_FFMPEG_EDITOR_H