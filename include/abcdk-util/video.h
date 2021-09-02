/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_VIDEO_H
#define ABCDK_UTIL_VIDEO_H

#include "abcdk-util/general.h"
#include "abcdk-util/ffmpeg.h"

__BEGIN_DECLS

#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H) && defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)


/** 视频捕获环境。*/
typedef struct _abcdk_video_capture  abcdk_video_capture_t;

/** 视频作者环境。*/
typedef struct _abcdk_video_writer  abcdk_video_writer_t;

/**
 * 关闭视频捕获环境。
*/
void abcdk_video_capture_close(abcdk_video_capture_t *vc);

/**
 * 查询视频中包含数据流的数量。
 * 
 * @return >= 0 成功(数量)，< 0 失败。
*/
int abcdk_video_capture_nb_streams(abcdk_video_capture_t *vc);

/**
 * 判断视频中数据流的类型。
 * 
 * @param stream_index 数据流索引。
 * @param type 1 视频，2 音频，3 字幕。
 * 
 * @return  0 是，!0 否。 
*/
int abcdk_video_capture_check_stream(abcdk_video_capture_t *vc,int stream_index,int type);

/**
 * 在视频中查找指定类型的数据流。
 * 
 * @param type 1 视频，2 音频，3 字幕。
 * 
 * @return  0 成功(数据流索引，以0为基值)，-1 失败(未找到)。 
*/
int abcdk_video_capture_find_stream(abcdk_video_capture_t *vc,int type);

/**
 * 获取视频中指定数据流的时长。
 * 
 * @return 秒.毫秒
*/
double abcdk_video_capture_get_duration(abcdk_video_capture_t *vc, int stream_index);

/**
 * 获取视频中指定数据流的FPS。
 * 
 * @return 秒.毫秒
*/
double abcdk_video_capture_get_fps(abcdk_video_capture_t *vc, int stream_index);

/**
 * 转换视频中指定数据流的TS时间到自然时间。
 * 
 * @return 秒.毫秒
*/
double abcdk_video_capture_ts2sec(abcdk_video_capture_t *vc, int stream_index, int64_t ts);

/**
 * 打开视频捕获环境。
 * 
 * @param timeout 超时(秒)。-1 直到有事件或出错。
 * @param dump !0 打印流信息，0 忽略。
*/
abcdk_video_capture_t *abcdk_video_capture_open(const char *short_name, const char *url, int64_t timeout, int dump);

/**
 * 读取数据包。
 * 
 * @param stream_index >=0 数据流索引，< 0 任意数据流。
 * @param only_key !0 仅读取关键帧，0 读取所有帧。
 * 
 * @return >= 0 成功(数据流索引)，< 0 失败。
*/
int abcdk_video_capture_read(abcdk_video_capture_t *vc, AVPacket *pkt, int stream_index, int only_key);

/**
 * 读取数据帧。
 * 
 * @param stream_index >=0 数据流索引，< 0 任意数据流。
 * @param only_key !0 仅读取关键帧，0 读取所有帧。
 * 
 * @return >= 0 成功(数据流索引)，< 0 失败。
*/
int abcdk_video_capture_read2(abcdk_video_capture_t *vc, AVFrame *fae, int stream_index, int only_key);


#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H && AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H


__END_DECLS

#endif //ABCDK_UTIL_FFMPEG2_H