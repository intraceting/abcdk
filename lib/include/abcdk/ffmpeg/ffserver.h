/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_FFMPEG_FFSERVER_H
#define ABCDK_FFMPEG_FFSERVER_H

#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/atomic.h"
#include "abcdk/shell/file.h"
#include "abcdk/ffmpeg/ffmpeg.h"

__BEGIN_DECLS

/**配置。 */
typedef struct _abcdk_ffserver_config
{
    /**源地址。*/
    const char *src_url;

    /**源格式。*/
    const char *src_fmt;

    /**源超时(秒)。*/
    int src_timeout;

    /** 源的速度(倍数)。0.01~100.0 */
    float src_speed;

    /** 源的最大延迟(秒.毫秒)。0.300~4.999*/
    float src_delay_max;

    /**源重试间隔(秒)。*/
    int src_retry;

    /**录像保存前缀。*/
    const char *record_prefix;

    /**录像分段时长。*/
    uint64_t record_duration;

    /**录像分段数量。*/
    uint16_t record_count;

    /**推流地址。*/
    const char *push_url;

    /**推流格式。*/
    const char *push_fmt;

    /**
     * 直播回调。
     * 
     * @warning 不能阻塞。
    */
    void (*live_cb)(void *opaque ,uint64_t id, const void *data, size_t size);

    /**直播环境指针。*/
    void *live_opaque;

    /**直播最大延时(秒.毫秒)。0.300~4.999。*/
    int live_delay_max;

    /**直播最大数量。*/
    int live_count_max;

}abcdk_ffserver_config_t;

/*简单的流媒体服务。*/
typedef struct _abcdk_ffserver abcdk_ffserver_t;

/**销毁。*/
void abcdk_ffserver_destroy(abcdk_ffserver_t **ctx);

/**
 * 创建。
 * 
 * @note 在服务关闭前，配置信息必须保持有效且不能更改。
 */
abcdk_ffserver_t *abcdk_ffserver_create(abcdk_ffserver_config_t *cfg);

/*停止。*/
void abcdk_ffserver_stop(abcdk_ffserver_t *ctx);

/*启动。*/
int abcdk_ffserver_start(abcdk_ffserver_t *ctx);

__END_DECLS

#endif //ABCDK_FFMPEG_FFSERVER_H