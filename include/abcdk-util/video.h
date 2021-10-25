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

/** 最大支持16个。*/
#define ABCDK_VIDEO_MAX_STREAMS     16

/**
 * 视频。
 * 
*/
typedef struct _abcdk_video
{
    /** 编/解码器。*/
    AVCodecContext *codec_ctx[ABCDK_VIDEO_MAX_STREAMS];

    /** 编/解码器字典。*/
    AVDictionary *codec_dict[ABCDK_VIDEO_MAX_STREAMS];

    /** 数据包过滤器。*/
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(58,20,100)
    AVBSFContext *vs_filter[ABCDK_VIDEO_MAX_STREAMS];
#else
    AVBitStreamFilterContext *vs_filter[ABCDK_VIDEO_MAX_STREAMS];
#endif

    /** 视频。*/
    AVFormatContext *ctx;

    /** 视频字典。*/
    AVDictionary *dict;

    /** 超时(秒)。*/
    int64_t timeout;

    /** 最近活动包时间(秒)。*/
    int64_t last_packet_time;

    /**
     * TS编号。
     * 
     * 0: PTS
     * 1: DTS
    */
    int64_t ts_nums[ABCDK_VIDEO_MAX_STREAMS][2];

} abcdk_video_t;

/**
 * 关闭视频。
*/
void abcdk_video_close(abcdk_video_t *video);

/**
 * 查询视频中包含数据流的数量。
 * 
 * @return >= 0 成功(数量)，< 0 失败。
*/
int abcdk_video_nb_streams(abcdk_video_t *video);

/**
 * 判断视频中数据流的类型。
 * 
 * @param stream_index 数据流索引。
 * @param type 1 视频，2 音频，3 字幕。
 * 
 * @return  0 是，!0 否。 
*/
int abcdk_video_check_stream(abcdk_video_t *video,int stream_index,int type);

/**
 * 在视频中查找指定类型的数据流。
 * 
 * @param type 1 视频，2 音频，3 字幕。
 * 
 * @return  0 成功(数据流索引，以0为基值)，-1 失败(未找到)。 
*/
int abcdk_video_find_stream(abcdk_video_t *video,int type);

/**
 * 获取视频中指定数据流的时长。
 * 
 * @return 秒.毫秒
*/
double abcdk_video_get_duration(abcdk_video_t *video, int stream_index);

/**
 * 获取视频中指定数据流图像的宽。
 * 
 * @return 像素
*/
int abcdk_video_get_width(abcdk_video_t *video, int stream_index);

/**
 * 获取视频中指定数据流图像的高。
 * 
 * @return 像素
*/
int abcdk_video_get_height(abcdk_video_t *video, int stream_index);

/**
 * 获取视频中指定数据流的FPS。
 * 
 * @return 秒.毫秒
*/
double abcdk_video_get_fps(abcdk_video_t *video, int stream_index);

/**
 * 转换视频中指定数据流TS时间到自然时间。
 * 
 * @return 秒.毫秒
*/
double abcdk_video_ts2sec(abcdk_video_t *video, int stream_index, int64_t ts);

/**
 * 打开视频(捕获)。
 * 
 * @param timeout 超时(秒)。-1 直到有事件或出错。
 * @param dump !0 打印流信息，0 忽略。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_video_t *abcdk_video_open_capture(const char *short_name, const char *url, int64_t timeout, int dump,
                                        const AVDictionary *dict);

/**
 * 读取数据包。
 * 
 * @param stream_index >=0 数据流索引，< 0 任意数据流。
 * @param only_key !0 仅读取关键帧，0 读取所有帧。
 * @param not_filter !0 禁用过滤器，0 启用过滤器。
 * 
 * @return >= 0 成功(数据流索引)，< 0 失败。
*/
int abcdk_video_read(abcdk_video_t *video, AVPacket *pkt, int stream_index, int only_key,int not_filter);

/**
 * 读取数据帧。
 * 
 * @param stream_index >=0 数据流索引，< 0 任意数据流。
 * @param only_key !0 仅读取关键帧，0 读取所有帧。
 * 
 * @return >= 0 成功(数据流索引)，< 0 失败。
*/
int abcdk_video_read2(abcdk_video_t *video, AVFrame *fae, int stream_index, int only_key);

/** 
 * 打开视频(作者)。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_video_t *abcdk_video_open_writer(const char*short_name,const char *url,const char *mime_type);

/**
 * 向视频中添加数据流。
 * 
 * @warning 仅支持视频流。
 * 
 * @param extdata 扩展数据的指针，NULL(0) 忽略。
 * @param extsize 扩展数据的长度。
 * @param have_codec 0 使用内部编码器(不支持B帧，并忽略外部扩展数据)，!0 使用外部编码器。
 * 
 * @return >= 0 成功(数据流索引)，< 0 失败。
*/
int abcdk_video_add_stream(abcdk_video_t *video, int fps, int width, int height, enum AVCodecID id,
                           const void *extdata, int extsize, int have_codec);

/**
 * 向视频中写入头部信息。
 * 
 * @param make_mp4fragment 0 默认的MP4格式，!0 支持片段的MP4格式。
 * @param dump !0 打印流信息，0 忽略。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_video_write_header(abcdk_video_t *video, int make_mp4fragment, int dump);

/**
 * 向视频中写入结束信息。
 * 
 * @note 自动写入所有延时编码数据包。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_video_write_trailer(abcdk_video_t *video);

/**
 * 向视频中写入数据包。
 * 
 * @note 如果是向网络流写入数据包，需要调用者限速调用此接口。因为过快的写入速度可能会造成网络数据包积压，连接被中断。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_video_write(abcdk_video_t *video, AVPacket *pkt);

/**
 * 向视频中写入数据帧。
 * 
 * @warning 仅支持图像。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_video_write2(abcdk_video_t *video, int stream_index, AVFrame *fae);

/**
 * 向视频中写入数据包。
 * 
 * @warning 不支持B帧。
 * 
 * @return >= 0 成功，< 0 失败。
*/
int abcdk_video_write3(abcdk_video_t *video, int stream_index, void *data, int size);

#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H && AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H


__END_DECLS

#endif //ABCDK_UTIL_FFMPEG2_H