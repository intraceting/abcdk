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

__BEGIN_DECLS

typedef struct _abcdk_ffmpeg_decoder abcdk_ffmpeg_decoder_t;

void abcdk_ffmpeg_decoder_free(abcdk_ffmpeg_decoder_t **ctx);
abcdk_ffmpeg_decoder_t *abcdk_ffmpeg_decoder_alloc();

int abcdk_ffmpeg_decoder_init(abcdk_ffmpeg_decoder_t *ctx, const AVCodec *codec_ctx, AVCodecParameters *param, int device);
int abcdk_ffmpeg_decoder_init2(abcdk_ffmpeg_decoder_t *ctx, const char *codec_name, AVCodecParameters *param, int device);
int abcdk_ffmpeg_decoder_init3(abcdk_ffmpeg_decoder_t *ctx, AVCodecID codec_id, AVCodecParameters *param, int device);


int abcdk_ffmpeg_decoder_recv(abcdk_ffmpeg_decoder_t *ctx, AVFrame *dst);
int abcdk_ffmpeg_decoder_send(abcdk_ffmpeg_decoder_t *ctx, AVPacket *src);

__END_DECLS

#endif // ABCDK_FFMPEG_DECODER_H