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
#include "abcdk/ffmpeg/util.h"

__BEGIN_DECLS

/**编码器环境.*/
typedef struct _abcdk_ffmpeg_encoder abcdk_ffmpeg_encoder_t;

/**释放.*/
void abcdk_ffmpeg_encoder_free(abcdk_ffmpeg_encoder_t **ctx);

/**创建.*/
abcdk_ffmpeg_encoder_t *abcdk_ffmpeg_encoder_alloc();

/**初始化.*/
int abcdk_ffmpeg_encoder_init(abcdk_ffmpeg_encoder_t *ctx, const AVCodec *codec_ctx, AVCodecParameters *param, int rate, int device);

/**初始化.*/
int abcdk_ffmpeg_encoder_init2(abcdk_ffmpeg_encoder_t *ctx, const char *codec_name, AVCodecParameters *param, int rate, int device);

/**初始化.*/
int abcdk_ffmpeg_encoder_init3(abcdk_ffmpeg_encoder_t *ctx, AVCodecID codec_id, AVCodecParameters *param, int rate, int device);

/**
 * 获取扩展数据.
 * 
 * @note 扩展数据由内部管理和释放, 外部只能引用.
 * 
 * @param [out] data 数据指针.
 * 
 * @return >=0 扩展数据长度. < 0 出错.
*/
int abcdk_ffmpeg_encoder_get_extradata(abcdk_ffmpeg_encoder_t *ctx, void **data);

/**
 * 接收.
 * 
 * @return > 0 有, 0 无, < 0 出错.
 */
int abcdk_ffmpeg_encoder_recv(abcdk_ffmpeg_encoder_t *ctx, AVPacket *dst);

/**
 * 发送. 
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_ffmpeg_encoder_send(abcdk_ffmpeg_encoder_t *ctx, AVFrame *src);

__END_DECLS

#endif // ABCDK_FFMPEG_ENCODER_H