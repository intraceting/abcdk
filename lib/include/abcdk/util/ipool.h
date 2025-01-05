/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_IPOOL_H
#define ABCDK_UTIL_IPOOL_H

#include "abcdk/util/general.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/object.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/rwlock.h"

__BEGIN_DECLS


/**简音的IP池。 */
typedef struct _abcdk_ipool abcdk_ipool_t;

/**销毁。 */
void abcdk_ipool_destroy(abcdk_ipool_t **ctx);

/**创建。 */
abcdk_ipool_t *abcdk_ipool_create();

/**
 * 重置。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_ipool_reset(abcdk_ipool_t *ctx, abcdk_sockaddr_t *begin, abcdk_sockaddr_t *end,
                      abcdk_sockaddr_t *dhcp_begin, abcdk_sockaddr_t *dhcp_end);

/**
 * 重置。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_ipool_reset2(abcdk_ipool_t *ctx, const char *begin, const char *end,
                       const char *dhcp_begin, const char *dhcp_end);

/**
 * 数量。 
 * 
 * @param flag 标志。0 全部，1 静态，2 动态。
*/
uint64_t abcdk_ipool_count(abcdk_ipool_t *ctx,int flag);

/**获取前缀长度(掩码)。*/
uint8_t abcdk_ipool_prefix(abcdk_ipool_t *ctx);

/**
 * 静态地址请求。
 * 
 * @return 0 成功，< 0 失败(已经被占用)。
*/
int abcdk_ipool_static_request(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr);

/**
 * 静态地址请求。
 * 
 * @return 0 成功，< 0 失败(已经被占用)。
*/
int abcdk_ipool_static_request2(abcdk_ipool_t *ctx,const char *addr);

/**
 * 动态地址请求。
 * 
 * @param [out] addr 地址。
 * 
 * @return 0 成功，< 0 失败(没有空闲的地址)。
*/
int abcdk_ipool_dhcp_request(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr);

/**
 * 回收。
 * 
 * @return 0 成功，< 0 失败(超出池范围)。
*/
int abcdk_ipool_reclaim(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr);

/** 
 * 验证地址。
 * 
 * @return 0 成功，< 0 失败(超出池范围)。
*/
int abcdk_ipool_verify(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr);

/**读锁。 */
void abcdk_ipool_rdlock(abcdk_ipool_t *ctx);

/**写锁。 */
void abcdk_ipool_wrlock(abcdk_ipool_t *ctx);

/**解锁。 */
int abcdk_ipool_unlock(abcdk_ipool_t *ctx,int exitcode);

__END_DECLS


#endif //ABCDK_UTIL_IPOOL_H