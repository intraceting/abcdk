/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_POOL_H
#define ABCDK_UTIL_POOL_H

#include "abcdk-util/general.h"
#include "abcdk-util/allocator.h"
#include "abcdk-util/thread.h"

__BEGIN_DECLS

/**
 * 一个简单的池子。
*/
typedef struct _abcdk_pool
{
    /**
     * 池子。
     * 
     * @note 尽量不要直接修改。
    */
    abcdk_allocator_t *table;

    /**
     * 队列长度。
     * 
     * @note 尽量不要直接修改。
    */
    size_t count;

    /**
     * 拉取游标。
     * 
     * @note 尽量不要直接修改。
    */
    size_t pull_pos;

    /**
     * 推送游标。
     * 
     * @note 尽量不要直接修改。
    */
    size_t push_pos;

} abcdk_pool_t;

/**
 * 销毁。
 * 
 * @warning 所有内存块必须全部在池内才能被销毁。
*/
void abcdk_pool_destroy(abcdk_pool_t *pool);

/**
 * 初始化。
 * 
 * @param size 大小。
 * @param number 数量。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_pool_init(abcdk_pool_t *pool, size_t size, size_t number);

/**
 * 拉取数据。
 * 
 * @return >= 0 成功(读取数据长度)，< 0 失败(空了)。
 * 
*/
ssize_t abcdk_pool_pull(abcdk_pool_t *pool, void *buf, size_t size);

/**
 * 推送数据。
 * 
 * @return >= 0 成功(写入数据长度)，< 失败(满了)。
 * 
*/
ssize_t abcdk_pool_push(abcdk_pool_t *pool, const void *buf, size_t size);

__END_DECLS

#endif //ABCDK_UTIL_POOL_H