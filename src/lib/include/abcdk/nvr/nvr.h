/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVR_H
#define ABCDK_NVR_H

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
#include "abcdk/rtsp/server.h"


__BEGIN_DECLS

/**简单的网络视频录像对象。*/
typedef struct _abcdk_nvr abcdk_nvr_t;

/**配置。 */
typedef struct _abcdk_nvr_config
{
    struct 
    {
        uint16_t port;

        const char *cert;
        const char *key;
    }rtsp;
    

}abcdk_nvr_config_t;

/**任务配置。 */
typedef struct _abcdk_nvr_task_config
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
        /**名字。*/
        const char *name;

    }play;

    struct
    {
        /**前缀。*/
        const char *prefix;

        /**分段时长(秒)。1~3600。*/
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
    } relay;

} abcdk_nvr_task_config_t;

/**销毁。*/
void abcdk_nvr_destroy(abcdk_nvr_t **ctx);

/**
 * 创建。
 *
 * @param rtsp_port 
 */
abcdk_nvr_t *abcdk_nvr_create(uint16_t rtsp_port);

/**
 * 删除任务。
 */
void abcdk_nvr_del_task(abcdk_nvr_t *ctx, uint64_t id);

/**
 * 添加任务。
 *
 * @return !0 成功，0 失败。
 */
uint64_t abcdk_nvr_add_task(abcdk_nvr_t *ctx, abcdk_nvr_task_config_t *cfg);




__END_DECLS

#endif // ABCDK_NVR_H