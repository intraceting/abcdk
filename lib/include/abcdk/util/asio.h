/*
 * This file is part of ABCDK.
 * 
 * MIT License
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

/**异步IO节点。*/
typedef struct _abcdk_asio_node abcdk_asio_node_t;

/**释放。 */
void abcdk_asio_unref(abcdk_asio_node_t **node_ctx);

/**引用。*/
abcdk_asio_node_t *abcdk_asio_refer(abcdk_asio_node_t *node_ctx);

/**申请。 */
abcdk_asio_node_t *abcdk_asio_alloc(abcdk_asio_t *ctx,size_t userdata, void (*free_cb)(void *userdata));

/**
 * 设置超时。
 * 
 * @note 看门狗精度为300毫秒。
 * 
 * @param [in] timeout 时长(毫秒)。> 0 启用， <= 0 禁用。
*/
void abcdk_asio_set_timeout(abcdk_asio_node_t *node_ctx, time_t timeout);

/**设置句柄。*/
void abcdk_asio_set_fd(abcdk_asio_node_t *node_ctx, int fd);

/**
 * 注册事件。
 * 
 * @param [in] fd 句柄。
 * @param [in] want 希望的事件。
 * @param [in] done 完成的事件。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_asio_mark(abcdk_asio_node_t *node_ctx,uint32_t want,uint32_t done);

/**销毁。*/
void abcdk_asio_destroy(abcdk_asio_t **ctx);

/**创建。*/
abcdk_asio_t *abcdk_asio_create();

/**
 * 等待事件。
 * 
 * @return > 0 事件数量。= 0 超时，< 0 失败。
*/
int abcdk_asio_wait(abcdk_asio_t *ctx,abcdk_epoll_event_t events[20],time_t timeout);





__END_DECLS

#endif //ABCDK_UTIL_ASIO_H