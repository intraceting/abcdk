/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_DHCP_IPOOL_H
#define ABCDK_DHCP_IPOOL_H

#include "abcdk/util/general.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/object.h"
#include "abcdk/util/socket.h"

/** IP池。 */
typedef struct _abcdk_ipool abcdk_ipool_t;

/**销毁。 */
void abcdk_ipool_destroy(abcdk_ipool_t **ctx);

/**创建。 */
abcdk_ipool_t *abcdk_ipool_create(abcdk_sockaddr_t *start,abcdk_sockaddr_t *end);

/**数量。 */
uint64_t abcdk_ipool_count(abcdk_ipool_t *ctx);

/**
 * 分配。
 * 
 * @return 0 成功，< 0 失败(没有空闲的地址)。
*/
int abcdk_ipool_allocate(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr);

/**
 * 回收。
 * 
 * @return 0 成功，< 0 失败(超出池范围)。
*/
int abcdk_ipool_reclaim(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr);

#endif //ABCDK_DHCP_IPOOL_H