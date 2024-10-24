/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_WRED_H
#define ABCDK_UTIL_WRED_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

/**
 * 简单的WRED(加权随机早期检测)算法。 
 * 
 * @note WRED：Weighted Random Early Detection
*/
typedef struct _abcdk_wred abcdk_wred_t;

/**销毁。*/
void abcdk_wred_destroy(abcdk_wred_t **ctx);

/**
 * 创建。
 * 
 * @param [in] min_th 最小阈值。
 * @param [in] max_th 最大阈值。
 * @param [in] weight 权重因子。
 * @param [in] prob 概率因子。
*/
abcdk_wred_t *abcdk_wred_create(int min_th,int max_th,int weight,int prob);

/**
 * 更新。
 * 
 * @param [in] qlen 队列长度。
 * 
 * @return 0 保留，!0 丢弃。
*/
int abcdk_wred_update(abcdk_wred_t *ctx,int qlen);


__END_DECLS

#endif // ABCDK_UTIL_WRED_H