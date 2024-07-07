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
#include "abcdk/util/bloom.h"
#include "abcdk/util/stream.h"
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

    /**源超时(秒)。-1~5。*/
    int src_timeout;

    /**源速度(倍数)。0.01~100.0 */
    float src_speed;

    /**源最大延迟(秒.毫秒)。0.300~4.999*/
    float src_delay_max;

    /**源重试间隔(秒)。1~30。*/
    int src_retry;

    /**
     * 录像前缀。
     * 
     * @note NULL(0) 禁用。
    */
    const char *record_prefix;

    /**录像分段时长。1~3600。*/
    int record_duration;

    /**录像分段数量。1~65535。*/
    int record_count;

    /**
     * 推流地址。
     * 
     * @note NULL(0) 禁用。
    */
    const char *push_url;

    /**推流格式。*/
    const char *push_fmt;

    /**直播最大延时(秒.毫秒)。0.300~4.999。*/
    float live_delay_max;

    /**直播最大连数量。1~99999。*/
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

/**停止。*/
void abcdk_ffserver_stop(abcdk_ffserver_t *ctx);

/**启动。*/
int abcdk_ffserver_start(abcdk_ffserver_t *ctx);

/**
 * 直播释放。
*/
void abcdk_ffserver_live_free(abcdk_ffserver_t *ctx,int id);

/**
 * 直播申请。
 * 
 * @return >= 0 成功(ID)，< 0 失败。
*/
int abcdk_ffserver_live_alloc(abcdk_ffserver_t *ctx);

/**
 * 直播拉流。 
 * 
 * @return >= 0 成功(长度)，< 0 失败。
 */
ssize_t abcdk_ffserver_live_fetch(abcdk_ffserver_t *ctx,int id ,void *buf,size_t size);

__END_DECLS

#endif //ABCDK_FFMPEG_FFSERVER_H