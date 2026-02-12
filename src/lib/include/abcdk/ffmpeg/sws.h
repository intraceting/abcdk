/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_SWS_H
#define ABCDK_FFMPEG_SWS_H

#include "abcdk/util/trace.h"
#include "abcdk/ffmpeg/ffmpeg.h"


__BEGIN_DECLS

/**格式转换环境.*/
typedef struct _abcdk_ffmpeg_sws abcdk_ffmpeg_sws_t;

/**销毁. */
void abcdk_ffmpeg_sws_free(abcdk_ffmpeg_sws_t **ctx);
/**创建. */
abcdk_ffmpeg_sws_t *abcdk_ffmpeg_sws_alloc();

/**
 * 转换.
 * 
 * @return >=0 成功, < 0 失败.
 */
int abcdk_ffmpeg_sws_scale(abcdk_ffmpeg_sws_t *ctx, const AVFrame *src, AVFrame *dst);

__END_DECLS

#endif //ABCDK_FFMPEG_SWS_H
