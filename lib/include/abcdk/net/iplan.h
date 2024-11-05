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

/**IP路径迭代器。 */
typedef struct _abcdk_iplan_iterator abcdk_iplan_iterator_t;

/**销毁。 */
void abcdk_iplan_destroy(abcdk_iplan_t **ctx);

/**
 * 创建。 
 * 
 * @param [in] have_port 包括端口。!0 是，0 否。
*/
abcdk_iplan_t *abcdk_iplan_create(int have_port);

/**
 * 删除。
 * 
 * @param [in] addrs IP地址。
 * 
 * @return 数据指针。
*/
void *abcdk_iplan_remove(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr);

/**
 * 插入。
 * 
 * @param [in] addrs IP地址。
 * @param [in] data 数据指针。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_iplan_insert(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr,void *data);

/**
 * 查询。
 * 
 * @param [in] addrs IP地址。
 * 
 * @return 数据指针。
*/
void *abcdk_iplan_lookup(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr);

/**
 * 遍历。
 * 
 * @param [in out] it 迭代器。NULL(0) 表示从头部开始遍历。
 * 
 * @return !NULL(0) 数据指针。NULL(0) 结束。
 */
void *abcdk_iplan_next(abcdk_iplan_t *ctx,abcdk_iplan_iterator_t **it);

/**读锁。 */
void abcdk_iplan_rdlock(abcdk_iplan_t *ctx);

/**写锁。 */
void abcdk_iplan_wrlock(abcdk_iplan_t *ctx);

/**解锁。 */
int abcdk_iplan_unlock(abcdk_iplan_t *ctx,int exitcode);


__END_DECLS

#endif //ABCDK_NET_IPLAN_H