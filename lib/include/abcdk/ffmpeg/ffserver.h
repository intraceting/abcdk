/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_FFMPEG_FFSERVER_H
#define ABCDK_FFMPEG_FFSERVER_H

#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/stream.h"
#include "abcdk/util/md5.h"
#include "abcdk/shell/file.h"
#include "abcdk/ffmpeg/ffmpeg.h"

__BEGIN_DECLS

#if defined(AVCODEC_AVCODEC_H) && defined(AVFORMAT_AVFORMAT_H) && defined(AVDEVICE_AVDEVICE_H)


/**简单的流媒体服务。*/
typedef struct _abcdk_ffserver abcdk_ffserver_t;

/**任务对象。*/
typedef struct _abcdk_ffserver_task abcdk_ffserver_task_t;

/**配置。 */
typedef struct _abcdk_ffserver_config
{
    /**
     * 标志。
     * 0：源，1：录像，2：推流，3：直播。
    */
    int flag;
#define ABCDK_FFSERVER_CFG_FLAG_SOURCE 0
#define ABCDK_FFSERVER_CFG_FLAG_RECORD 1
#define ABCDK_FFSERVER_CFG_FLAG_PUSH 2
#define ABCDK_FFSERVER_CFG_FLAG_LIVE 3

    /**提示。*/
    const char *tip;

    union
    {

        struct
        {

            /**地址。*/
            const char *url;

            /**格式。*/
            const char *fmt;

            /**超时(秒)。-1~5。*/
            int timeout;

            /**速度(倍数)。0.01~100.0 */
            float speed;

            /**最大延迟(秒.毫秒)。0.300~4.999*/
            float delay_max;

            /**源重试间隔(秒)。1~30。*/
            int retry;
        } src;

        struct
        {

            /**前缀。*/
            const char *prefix;

            /**分段时长。1~3600。*/
            int duration;

            /**分段数量。1~65535。*/
            int count;
        } record;

        struct
        {

            /**地址。*/
            const char *url;

            /**格式。*/
            const char *fmt;
        } push;

        struct
        {
            /**流缓存。*/
            abcdk_stream_t *buf;

            /**最大延时(秒.毫秒)。0.300~4.999。*/
            float delay_max;

            /** 
             * 缓存更新通知。
             * 
             * @note NULL(0) 忽略。
             * @warning 不能被阻塞。
            */
            void (*ready_cb)(void *opaque);

            /**
             * 任务移除通知。
             * 
             * @note NULL(0) 忽略。
             * @warning 不能被阻塞。
            */
            void (*delete_cb)(void *opaque);

            /**环境指针。*/
            void *opaque;

        } live;

    } u;
} abcdk_ffserver_config_t;


/**销毁。*/
void abcdk_ffserver_destroy(abcdk_ffserver_t **ctx);

/**
 * 创建。
 *
 * @note 在服务关闭前，配置信息必须保持有效且不能更改。
 */
abcdk_ffserver_t *abcdk_ffserver_create(abcdk_ffserver_config_t *cfg);

/**
 * 心跳。
*/
void abcdk_ffserver_task_heartbeat(abcdk_ffserver_t *ctx, abcdk_ffserver_task_t *task);

/**
 * 获取索引。
 * 
 * @note 进程内唯一。
 * 
 */
uint64_t abcdk_ffserver_get_index(abcdk_ffserver_t *ctx,abcdk_ffserver_task_t *task);

/**
 * 删除任务。
 */
void abcdk_ffserver_task_del(abcdk_ffserver_t *ctx, abcdk_ffserver_task_t **task);

/**
 * 添加任务。
 *
 * @return >= 0 成功(ID)，< 0 失败。
 */
abcdk_ffserver_task_t *abcdk_ffserver_task_add(abcdk_ffserver_t *ctx,abcdk_ffserver_config_t *cfg);


#endif //AVCODEC_AVCODEC_H && AVFORMAT_AVFORMAT_H && AVDEVICE_AVDEVICE_H


__END_DECLS

#endif // ABCDK_FFMPEG_FFSERVER_H