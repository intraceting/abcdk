/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_DECODER_H
#define ABCDK_FFMPEG_DECODER_H

#include "abcdk/util/trace.h"
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/sws.h"
#include "abcdk/ffmpeg/util.h"

__BEGIN_DECLS

/**解码器环境.*/
typedef struct _abcdk_ffmpeg_decoder abcdk_ffmpeg_decoder_t;

/**释放.*/
void abcdk_ffmpeg_decoder_free(abcdk_ffmpeg_decoder_t **ctx);

/**创建.*/
abcdk_ffmpeg_decoder_t *abcdk_ffmpeg_decoder_alloc();

/**初始化.*/
int abcdk_ffmpeg_decoder_init(abcdk_ffmpeg_decoder_t *ctx, const AVCodec *codec_ctx, AVCodecParameters *param, int device);

/**初始化.*/
int abcdk_ffmpeg_decoder_init2(abcdk_ffmpeg_decoder_t *ctx, const char *codec_name, AVCodecParameters *param, int device);

/**初始化.*/
int abcdk_ffmpeg_decoder_init3(abcdk_ffmpeg_decoder_t *ctx, AVCodecID codec_id, AVCodecParameters *param, int device);

/**
 * 接收.
 * 
 * @return > 0 有, 0 无, < 0 出错.
 */
int abcdk_ffmpeg_decoder_recv(abcdk_ffmpeg_decoder_t *ctx, AVFrame *dst);

/**
 * 发送. 
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_ffmpeg_decoder_send(abcdk_ffmpeg_decoder_t *ctx, AVPacket *src);

__END_DECLS

#endif // ABCDK_FFMPEG_DECODER_H