/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_NET_IPLAN_H
#define ABCDK_NET_IPLAN_H

#include "abcdk/util/general.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/object.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/map.h"
#include "abcdk/util/rwlock.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS


/**简单的IP路径。 */
typedef struct _abcdk_iplan abcdk_iplan_t;

/**配置。*/
typedef struct _abcdk_iplan_config
{
    /**包括端口。!0 是，0 否。*/
    int have_port;

    /**环境指针。*/
    void *opaque;

    /**删除回调函数。*/
    void (*remove_cb)(abcdk_sockaddr_t *addr,void *userdata, void *opaque);
    
}abcdk_iplan_config_t;

/**销毁。 */
void abcdk_iplan_destroy(abcdk_iplan_t **ctx);

/**创建。 */
abcdk_iplan_t *abcdk_iplan_create(abcdk_iplan_config_t *cfg);

/**删除。*/
void abcdk_iplan_remove(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr);

/**
 * 查询。
 * 
 * @note 如果不存在，则自动创建。
 * 
 * @param [in] userdata 用户环境大小。= 0 仅查询。
 * 
 * @return !NULL(0) 成功(用户环境指针)，NULL(0) 失败。
*/
void *abcdk_iplan_lookup(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr,size_t userdata);

/**
 * 遍历。
 * 
 * @param [in out] it 迭代器。NULL(0) 表示从头部开始遍历。
 * 
 * @return !NULL(0) 数据指针。NULL(0) 结束。
 */
void *abcdk_iplan_next(abcdk_iplan_t *ctx,void **it);

/**读锁。 */
void abcdk_iplan_rdlock(abcdk_iplan_t *ctx);

/**写锁。 */
void abcdk_iplan_wrlock(abcdk_iplan_t *ctx);

/**解锁。 */
int abcdk_iplan_unlock(abcdk_iplan_t *ctx,int exitcode);


__END_DECLS

#endif //ABCDK_NET_IPLAN_H