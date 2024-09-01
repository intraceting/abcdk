/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_CONTEXT_H
#define ABCDK_UTIL_CONTEXT_H

#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/object.h"
#include "abcdk/util/io.h"
#include "abcdk/util/mutex.h"
#include "abcdk/util/spinlock.h"
#include "abcdk/util/rwlock.h"

__BEGIN_DECLS

/**简单的上下文环境。 */
typedef struct _abcdk_context abcdk_context_t;

/**常量。*/
typedef enum _abcdk_context_constant
{
    /**无。*/
    ABCDK_CONTEXT_SYNC_NON = 0,
#define ABCDK_CONTEXT_SYNC_NON   ABCDK_CONTEXT_SYNC_NON

    /**互斥量。*/
    ABCDK_CONTEXT_SYNC_MUTEX = 1,
#define ABCDK_CONTEXT_SYNC_MUTEX   ABCDK_CONTEXT_SYNC_MUTEX

    /**自旋锁。*/
    ABCDK_CONTEXT_SYNC_SPINLOCK = 2,
#define ABCDK_CONTEXT_SYNC_SPINLOCK   ABCDK_CONTEXT_SYNC_SPINLOCK

    /** 读写锁。*/
    ABCDK_CONTEXT_SYNC_RWLOCK = 3,
#define ABCDK_CONTEXT_SYNC_RWLOCK   ABCDK_CONTEXT_SYNC_RWLOCK

    /** 读写描述符。*/
    ABCDK_CONTEXT_FD_RW = 0,
#define ABCDK_CONTEXT_FD_RW   ABCDK_CONTEXT_FD_RW

    /** 读描述符。*/
    ABCDK_CONTEXT_FD_RD = 1,
#define ABCDK_CONTEXT_FD_RD   ABCDK_CONTEXT_FD_RD

    /** 写描述符。*/
    ABCDK_CONTEXT_FD_WR = 2,
#define ABCDK_CONTEXT_FD_WR   ABCDK_CONTEXT_FD_WR

}abcdk_context_constant_t;

/**释放。*/
void abcdk_context_unref(abcdk_context_t **ctx);

/**引用。*/
abcdk_context_t *abcdk_context_refer(abcdk_context_t *src);

/**
 * 申请。
 * 
 * @param [in] sync_type 锁类型。
 * @param [in] user_data 用户数据长度。
 * @param [in] free_cb 用户数据销毁函数。
 * 
*/
abcdk_context_t *abcdk_context_alloc(int sync_type, size_t userdata, void (*free_cb)(void *userdata));

/** 获取用户环境指针。*/
void *abcdk_context_get_userdata(abcdk_context_t *ctx);

/**
 * 设置描述符。
 * 
 * @param [in] flag 标志。
 * 
*/
void abcdk_context_set_fd(abcdk_context_t *ctx,int fd, int flag);

/**
 * 获取描述符。
 * 
 * @return >=0 成功(旧的句柄)，< 0  失败(未设置或读写句柄不一致)。
*/
int abcdk_context_get_fd(abcdk_context_t *ctx,int flag);

/**解锁。 */
void abcdk_context_unlock(abcdk_context_t *ctx);

/**加锁。 */
void abcdk_context_lock(abcdk_context_t *ctx);

/**读锁。 */
void abcdk_context_rdlock(abcdk_context_t *ctx);

/**写锁。 */
void abcdk_context_wrlock(abcdk_context_t *ctx);

/**通知。 */
void abcdk_context_signal(abcdk_context_t *ctx,int broadcast);

/**
 * 等待。
 * 
 * @param [in] timeout – 超时(毫秒)。< 0 直到有事件或出错。
 * 
 * @return 0 成功(有事件)，!0 失败(超时或出错)。
 */
int abcdk_context_wait(abcdk_context_t *ctx, time_t timeout);


__END_DECLS

#endif //ABCDK_UTIL_CONTEXT_H
