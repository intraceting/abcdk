/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_EPOLLEX_H
#define ABCDK_UTIL_EPOLLEX_H

#include "abcdk/util/map.h"
#include "abcdk/util/pool.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/clock.h"
#include "abcdk/util/epoll.h"
#include "abcdk/util/socket.h"

__BEGIN_DECLS

/** epoll扩展对象。*/
typedef struct _abcdk_epollex abcdk_epollex_t;

/** 清理回调函数。*/
typedef void (*abcdk_epollex_cleanup_cb)(epoll_data_t *data, void *opaque);

/**
 * 销毁环境。
*/
void abcdk_epollex_free(abcdk_epollex_t **ctx);

/**
 * 创建环境。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_epollex_t *abcdk_epollex_alloc(abcdk_epollex_cleanup_cb cleanup_cb, void *opaque);

/**
 * 分离句柄。
 * 
 * @note 关联成功后，句柄在分离前不可被关闭或释放。
 * 
 * @return 0 成功，!0 失败(或不存在)。
*/
int abcdk_epollex_detach(abcdk_epollex_t *ctx,int fd);

/**
 * 关联句柄。
 * 
 * @note 关联成功后，句柄在分离前不可被关闭或释放。
 * @note 默认空闲超时30秒。
 * 
 * @param [in] fd 句柄。
 * @param [in] data 关联数据。
 * 
 * @return 0 成功，!0 失败(或重复)。
*/
int abcdk_epollex_attach(abcdk_epollex_t *ctx, int fd, const epoll_data_t *data);

/**
 * 关联句柄。
 * 
 * @return 0 成功，!0 失败(或重复)。
*/
int abcdk_epollex_attach2(abcdk_epollex_t *ctx, int fd);

/** 
 * 统计关联句柄数量。
 * 
*/
size_t abcdk_epollex_count(abcdk_epollex_t *ctx);

/**
 * 设置超时。
 * 
 * @note 1、看门狗精度为1000毫秒；2、超时生效时间受引擎的工作周期影响。
 * 
 * @param [in] timeout 超时(毫秒)。
 * 
 * @return 0 成功，!0 失败(或不存在)。
*/
int abcdk_epollex_timeout(abcdk_epollex_t *ctx, int fd,time_t timeout);

/**
 * 注册事件。
 * 
 * @param [in] fd 句柄，-1 广播到所有句柄。
 * @param [in] want 希望的事件。
 * @param [in] done 完成的事件，广播无效。
 * 
 * @return 0 成功，!0 失败(或不存在)。
*/
int abcdk_epollex_mark(abcdk_epollex_t *ctx,int fd,uint32_t want,uint32_t done);

/**
 * 等待事件。
 * 
 * @param [in] timeout 超时(毫秒)。
 * 
 * @return 0 成功，< 0 失败(或超时)。
*/
int abcdk_epollex_wait(abcdk_epollex_t *ctx,abcdk_epoll_event_t *event,time_t timeout);

/**
 * 引用释放。
 * 
 * @return 0 成功，!0 失败(或不存在)。
*/
int abcdk_epollex_unref(abcdk_epollex_t *ctx,int fd, uint32_t events);



__END_DECLS

#endif //ABCDK_UTIL_EPOLLEX_H