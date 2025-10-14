/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_BSF_H
#define ABCDK_FFMPEG_BSF_H

#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"
#include "abcdk/ffmpeg/ffmpeg.h"

__BEGIN_DECLS

/**流过滤环境. */
typedef struct _abcdk_ffmpeg_bsf abcdk_ffmpeg_bsf_t;

/**销毁. */
void abcdk_ffmpeg_bsf_free(abcdk_ffmpeg_bsf_t **ctx);
/**创建. */
abcdk_ffmpeg_bsf_t *abcdk_ffmpeg_bsf_alloc(const char *name);

/**
 * 初始化. 
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_ffmpeg_bsf_init(abcdk_ffmpeg_bsf_t *ctx, const AVCodecContext *codec_ctx);

/**
 * 初始化. 
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_ffmpeg_bsf_init2(abcdk_ffmpeg_bsf_t *ctx, const AVCodecParameters *codec_par);

/**
 * 接收. 
 * 
 * @return > 0 有, 0 无, < 0 失败.
*/
int abcdk_ffmpeg_bsf_recv(abcdk_ffmpeg_bsf_t *ctx, AVPacket *pkt);

/**
 * 发送. 
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_ffmpeg_bsf_send(abcdk_ffmpeg_bsf_t *ctx, AVPacket *pkt);

__END_DECLS

#endif //ABCDK_FFMPEG_BSF_H
