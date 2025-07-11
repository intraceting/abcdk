/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_FFMPEG_FFRCORD_H
#define ABCDK_FFMPEG_FFRCORD_H

#include "abcdk/util/time.h"
#include "abcdk/util/path.h"
#include "abcdk/ffmpeg/ffeditor.h"

__BEGIN_DECLS

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)

/**简单的视音录像。*/
typedef struct _abcdk_ffrecord abcdk_ffrecord_t;

/**销毁。 */
void abcdk_ffrecord_destroy(abcdk_ffrecord_t **ctx);

/**创建。*/
abcdk_ffrecord_t *abcdk_ffrecord_create(const char *save_path,const char *segment_prefix, int segment_duration,int segment_size);

/**
 * 添加流。
 * 
 * @return  0 成功，< 0 失败(已满或重复)。
*/
int abcdk_ffrecord_add_stream(abcdk_ffrecord_t *ctx, AVStream *src_stream);

/**
 * 写数据包。
 * 
 * @note 视频流关键帧标志必须周期性存在，否则无法正常分段。
 * 
 * @param [in] src_pkt 源数据包。NULL(0) 结束。
 *
 * @return  0 成功，< 0 失败(已满或其它)。
 */
int abcdk_ffrecord_write_packet(abcdk_ffrecord_t *ctx, AVPacket *src_pkt, AVRational *src_time_base);

/**
 * 写入数据包。
 * 
 * @note 仅支持视频流，并且不支持B帧。
 * @note 视频流关键帧标志必须周期性存在，否则无法正常分段。
 * 
 * @return 0 成功，< 0 失败(已满或其它)。
*/
int abcdk_ffrecord_write_packet2(abcdk_ffrecord_t *ctx, void *data, int size, int keyframe, int stream);

#endif //AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H

__END_DECLS

#endif //ABCDK_FFMPEG_FFRCORD_H