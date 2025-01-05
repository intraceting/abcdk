/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_ASIOEX_H
#define ABCDK_UTIL_ASIOEX_H

#include "abcdk/util/asio.h"


__BEGIN_DECLS

/**异步IO对象扩展。*/
typedef struct _abcdk_asioex  abcdk_asioex_t;

/**销毁。*/
void abcdk_asioex_destroy(abcdk_asioex_t **ctx);

/**
 * 创建。
 * 
 * @param [in] group 分组数量。
 * @param [in] max 最大数量。
*/
abcdk_asioex_t *abcdk_asioex_create(int group,int max);

/**取消等除。 */
void abcdk_asioex_abort(abcdk_asioex_t *ctx);

/** 
 * 分派。
 * 
 * @note 分派的对象不能被销毁。
 * 
 * @param [in] idx 索引。< 0 自由分配。
*/
abcdk_asio_t *abcdk_asioex_dispatch(abcdk_asioex_t *ctx,int idx);

__END_DECLS

#endif //ABCDK_UTIL_ASIO_H