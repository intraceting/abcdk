/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_ENCODER_H
#define ABCDK_FFMPEG_ENCODER_H

#include "abcdk/util/trace.h"
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/sws.h"

__BEGIN_DECLS

typedef struct _abcdk_ffmpeg_encoder abcdk_ffmpeg_encoder_t;

void abcdk_ffmpeg_encoder_free(abcdk_ffmpeg_encoder_t **ctx);
abcdk_ffmpeg_encoder_t *abcdk_ffmpeg_encoder_alloc();

int abcdk_ffmpeg_encoder_init(abcdk_ffmpeg_encoder_t *ctx, const AVCodec *codec_ctx, AVCodecParameters *param, int framerate, int device);
int abcdk_ffmpeg_encoder_init2(abcdk_ffmpeg_encoder_t *ctx, const char *codec_name, AVCodecParameters *param, int framerate, int device);
int abcdk_ffmpeg_encoder_init3(abcdk_ffmpeg_encoder_t *ctx, AVCodecID codec_id, AVCodecParameters *param, int framerate, int device);

int abcdk_ffmpeg_encoder_get_extradata(abcdk_ffmpeg_encoder_t *ctx, void **data);

int abcdk_ffmpeg_encoder_recv(abcdk_ffmpeg_encoder_t *ctx, AVPacket *dst);
int abcdk_ffmpeg_encoder_send(abcdk_ffmpeg_encoder_t *ctx, AVFrame *src);

__END_DECLS

#endif // ABCDK_FFMPEG_ENCODER_H