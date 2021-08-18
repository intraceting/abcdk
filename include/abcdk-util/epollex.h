/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_EPOLLEX_H
#define ABCDK_UTIL_EPOLLEX_H

#include "abcdk-util/map.h"
#include "abcdk-util/pool.h"
#include "abcdk-util/thread.h"
#include "abcdk-util/clock.h"
#include "abcdk-util/epoll.h"
#include "abcdk-util/socket.h"

__BEGIN_DECLS

/** epoll扩展对象。*/
typedef struct _abcdk_epollex abcdk_epollex_t;

/**
 * 销毁多路复用器环境。
*/
void abcdk_epollex_free(abcdk_epollex_t **ctx);

/**
 * 创建多路复用器环境。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_epollex_t *abcdk_epollex_alloc();

/**
 * 分离句柄。
 * 
 * @warning 关联成功后，句柄在分离前不可被关闭或释放。
 * 
 * @return 0 成功，!0 失败(或不存在)。
*/
int abcdk_epollex_detach(abcdk_epollex_t *ctx,int fd);

/**
 * 关联句柄。
 * 
 * @warning 关联成功后，句柄在分离前不可被关闭或释放。
 * 
 * @param fd 句柄。
 * @param data 关联数据。
 * @param timeout 超时(毫秒)，<=0 忽略。
 * 
 * @return 0 成功，!0 失败(或重复)。
*/
int abcdk_epollex_attach(abcdk_epollex_t *ctx, int fd, const epoll_data_t *data,time_t timeout);

/**
 * 关联句柄。
 * 
 * @warning 关联成功后，句柄在分离前不可被关闭或释放。
 * 
 * @return 0 成功，!0 失败(或重复)。
*/
int abcdk_epollex_attach2(abcdk_epollex_t *ctx, int fd,time_t timeout);

/**
 * 注册事件。
 * 
 * @param fd 句柄，-1 广播到所有句柄。
 * @param want 希望的事件。
 * @param done 完成的事件，广播无效。
 * 
 * @return 0 成功，!0 失败(或不存在)。
*/
int abcdk_epollex_mark(abcdk_epollex_t *ctx,int fd,uint32_t want,uint32_t done);

/**
 * 等待事件。
 * 
 * @param timeout 超时(毫秒)。>= 0 有事件或时间过期，< 0 直到有事件或出错。
 * 
 * @return >=0 成功，!0 失败(或超时)。
*/
int abcdk_epollex_wait(abcdk_epollex_t *ctx,abcdk_epoll_event *event,time_t timeout);

/**
 * 引用释放。
 * 
 * @return 0 成功，!0 失败(或不存在)。
*/
int abcdk_epollex_unref(abcdk_epollex_t *ctx,int fd, uint32_t events);


__END_DECLS

#endif //ABCDK_UTIL_EPOLLEX_H