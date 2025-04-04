/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_FFMPEG_NVR_H
#define ABCDK_FFMPEG_NVR_H

#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/stream.h"
#include "abcdk/util/md5.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/registry.h"
#include "abcdk/system/file.h"
#include "abcdk/ffmpeg/ffeditor.h"


__BEGIN_DECLS

/**简单的网络视频录像对象。*/
typedef struct _abcdk_ffmpeg_nvr abcdk_ffmpeg_nvr_t;

/**配置。 */
typedef struct _abcdk_ffmpeg_nvr_config
{
    /**
     * 标志。
     * 0：源，1：录像，2：推流，3：直播。
    */
    int flag;
#define ABCDK_FFMPEG_NVR_CFG_FLAG_SRC   0
#define ABCDK_FFMPEG_NVR_CFG_FLAG_REC   1
#define ABCDK_FFMPEG_NVR_CFG_FLAG_PUSH  2
#define ABCDK_FFMPEG_NVR_CFG_FLAG_LIVE  3

    /**提示。*/
    const char *tip;

    /** 应用环境指针。*/
    void *opaque;

    /**
     * 移除通知。
     *
     * @note NULL(0) 忽略。
     * @warning 不能被阻塞。
     */
    void (*remove_cb)(void *opaque);

    union
    {

        struct
        {
            /**
             * 地址。
             * 
             * @note unix://本地管道。
            */
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

            /** 
             * 缓存更新通知。
             * 
             * @note NULL(0) 忽略。
             * @warning 不能被阻塞。
            */
            void (*ready_cb)(void *opaque);

            /**最大延时(秒.毫秒)。0.300~4.999。*/
            float delay_max;

        } live;

    } u;
} abcdk_ffmpeg_nvr_config_t;

/**销毁。*/
void abcdk_ffmpeg_nvr_destroy(abcdk_ffmpeg_nvr_t **ctx);

/**
 * 创建。
 *
 * @note 在服务关闭前，配置信息必须保持有效且不能更改。
 */
abcdk_ffmpeg_nvr_t *abcdk_ffmpeg_nvr_create(abcdk_ffmpeg_nvr_config_t *cfg);

/**
 * 删除任务。
 */
void abcdk_ffmpeg_nvr_task_del(abcdk_ffmpeg_nvr_t *ctx, const char *id);

/**
 * 添加任务。
 *
 * @return 0 成功，- 1 失败。
 */
int abcdk_ffmpeg_nvr_task_add(abcdk_ffmpeg_nvr_t *ctx, const char *id, abcdk_ffmpeg_nvr_config_t *cfg);

/**
 * 心跳。
*/
void abcdk_ffmpeg_nvr_task_heartbeat(abcdk_ffmpeg_nvr_t *ctx, const char *id);



__END_DECLS

#endif // ABCDK_FFMPEG_NVR_H