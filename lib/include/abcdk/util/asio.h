/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_ASIO_H
#define ABCDK_UTIL_ASIO_H

#include "abcdk/util/map.h"
#include "abcdk/util/pool.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/clock.h"
#include "abcdk/util/epoll.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/mutex.h"
#include "abcdk/util/spinlock.h"
#include "abcdk/util/rwlock.h"
#include "abcdk/util/context.h"

__BEGIN_DECLS

/**异步IO对象。*/
typedef struct _abcdk_asio  abcdk_asio_t;

/**销毁。*/
void abcdk_asio_destroy(abcdk_asio_t **ctx);

/**
 * 创建。
 * 
 * @param [in] max 最大数量。
*/
abcdk_asio_t *abcdk_asio_create(int max);

/**获取数量。 */
size_t abcdk_asio_count(abcdk_asio_t *ctx);

/** 
 * 解绑句柄。
 * 
 * @warning 添加的句柄(真实)由创建者负责关闭。
 * 
 * @return 0 成功。< 0 失败(不存在)。
*/
int abcdk_asio_detch(abcdk_asio_t *ctx,int64_t pfd);

/**
 * 绑定句柄。
 * 
 * @param [in] fd 句柄(真实)。
 * @param [in] userdata 用户数据。
 *
 * @return > 0 成功(伪句柄)，<= 0 失败。
 */
int64_t abcdk_asio_attach(abcdk_asio_t *ctx, int fd, epoll_data_t *userdata);

/**
 * 设置超时。
 * 
 * @note 看门狗精度为1秒。
 * 
 * @param [in] pfd 伪句柄。
 * @param [in] timeout 时长(秒)。!0 有效，0 禁用。默认：0。
 * 
 * @return 0 成功。< 0 失败(不存在)。
*/
int abcdk_asio_timeout(abcdk_asio_t *ctx,int64_t pfd, time_t timeout);

/**
 * 注册事件。
 * 
 * @param [in] pfd 伪句柄。
 * @param [in] want 希望的事件。
 * @param [in] done 完成的事件。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_asio_mark(abcdk_asio_t *ctx,int64_t pfd,uint32_t want,uint32_t done);

/**
 * 释放。
 * 
 * @param [in] pfd 伪句柄。
 * 
 * @return 0 成功，< 0 失败(不存在)。
*/
int abcdk_asio_unref(abcdk_asio_t *ctx,int64_t pfd, uint32_t events);

/**
 * 等待事件。
 * 
 * @note 仅允许固定线程调用此接口。
 * 
 * @return > 0 有事件，= 0 无事件，< 0 出错。
*/
int abcdk_asio_wait(abcdk_asio_t *ctx,abcdk_epoll_event_t *event);

/**
 * 取消等待。
 * 
 * @note 所有关联的句柄都将收到错误事件。
*/
void abcdk_asio_abort(abcdk_asio_t *ctx);

__END_DECLS

#endif //ABCDK_UTIL_ASIO_H