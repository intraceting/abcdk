/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_NONCE_H
#define ABCDK_UTIL_NONCE_H

#include "abcdk/util/general.h"
#include "abcdk/util/registry.h"
#include "abcdk/util/timer.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/random.h"

__BEGIN_DECLS

/**简单的NONCE环境。 */
typedef struct _abcdk_nonce abcdk_nonce_t;

/**销毁*/
void abcdk_nonce_destroy(abcdk_nonce_t **ctx);

/**创建。*/
abcdk_nonce_t *abcdk_nonce_create();

/**
 * 重置。
 *
 * @param [in] time_diff 时间误差(毫秒)。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_nonce_reset(abcdk_nonce_t *ctx,uint64_t time_diff);

/**
 * 生成。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_nonce_generate(abcdk_nonce_t *ctx,uint8_t key[32]);

/**
 * 检查。
 * 
 * @return 0 有效，-1 过期，-2 重复。
*/
int abcdk_nonce_check(abcdk_nonce_t *ctx,const uint8_t key[32]);


__END_DECLS

#endif //ABCDK_UTIL_NONCE_H