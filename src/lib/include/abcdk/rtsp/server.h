/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_H
#define ABCDK_RTSP_SERVER_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/rtsp/rtsp.h"

__BEGIN_DECLS

/**简单的RTSP服务。*/
typedef struct _abcdk_rtsp_server abcdk_rtsp_server_t;

/**销毁。*/
void abcdk_rtsp_server_destroy(abcdk_rtsp_server_t **ctx);

/**
 * 创建。
 * 
 * @param [in] flag 标志。0x01(IPV4)，0x02(IPV6)，0x10(UDP)。
*/
abcdk_rtsp_server_t *abcdk_rtsp_server_create(uint16_t port, int flag);

/**
 * 设置授权管理。
 * 
 * @param [in] realm 领域。NULL(0) 默认。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_rtsp_server_set_auth(abcdk_rtsp_server_t *ctx,const char  *realm);

/**
 * 设置TLS证书和私钥。
 * 
 * @param [in] enable_srtp 启用SRTP协议。0 禁用，!0 启用。
 * @param [in] encrypt_srtp SRTP数据加密。0 否，!0 是。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_rtsp_server_set_tls(abcdk_rtsp_server_t *ctx,const char *cert,const char *key, int enable_srtp, int encrypt_srtp);

/**停止。*/
void abcdk_rtsp_server_stop(abcdk_rtsp_server_t *ctx);

/**启动。*/
int abcdk_rtsp_server_start(abcdk_rtsp_server_t *ctx);

/**删除账户。*/
void abcdk_rtsp_server_remove_user(abcdk_rtsp_server_t *ctx, const char *username);

/**
 * 添加账户。
 *
 * @note 如果账户已经存在，则只更新密码。
 * 
 * @param [in] scheme 密钥方案。
 * @param [in] totp_time_step 时间步长(秒)。
 * @param [in] totp_digit_size 数字数量(个)。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_rtsp_server_add_user(abcdk_rtsp_server_t *ctx, const char *username, const char *password, int scheme, int totp_time_step, int totp_digit_size);

/**删除媒体。*/
void abcdk_rtsp_server_remove_media(abcdk_rtsp_server_t *ctx, const char *name);

/**
 * 播放媒体。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_rtsp_server_play_media(abcdk_rtsp_server_t *ctx, const char *name);

/**
 * 创建媒体。
 * 
 * @param [in] name 资源名称。
 * @param [in] title 标题。NULL(0) 默认。
 * @param [in] comment 注释。NULL(0) 默认。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_rtsp_server_create_media(abcdk_rtsp_server_t *ctx, const char *name, const char *title, const char *comment);

/**
 * 向媒体添加流。
 * 
 * @param [in] extdata 编码信息。函数返回后，对象可以立即删除或修改。
 * @param [in] bitrate 码率(kbps)。
 * @param [in] cache 缓存(帧)。
 * 
 * @return > 0 成功(流ID)，<= 0 失败。
*/
int abcdk_rtsp_server_add_stream(abcdk_rtsp_server_t *ctx, const char *name, int codec, abcdk_object_t *extdata, uint32_t bitrate, int cache);

/**
 * 向媒体播放流。
 * 
 * @param [in] pts 显示时间(微秒)。
 * @param [in] dur 播放时长(微秒)。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_rtsp_server_play_stream(abcdk_rtsp_server_t *ctx, const char *name, int stream, const void *data, size_t size, int64_t pts, int64_t dur);


__END_DECLS

#endif //ABCDK_RTSP_SERVER_H