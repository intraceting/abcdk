/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "util/pool.h"

void abcdk_pool_destroy(abcdk_pool_t *pool)
{
    assert(pool != NULL);

    abcdk_object_unref(&pool->table);

    memset(pool, 0, sizeof(*pool));
}

int abcdk_pool_init(abcdk_pool_t *pool, size_t size, size_t number)
{
    assert(pool != NULL && size > 0 && number > 0);

    pool->table = abcdk_object_alloc(&size, number, 1);
    if (!pool->table)
        return -1;

    pool->count = 0;
    pool->pull_pos = 0;
    pool->push_pos = 0;

    return 0;
}

int abcdk_pool_pull(abcdk_pool_t *pool, void *buf)
{
    assert(pool != NULL && buf != NULL);

    /*池不能是空的。*/
    if (pool->count > 0)
    {
        /*按游标位置从池子中读取数据。*/
        memcpy(buf, pool->table->pptrs[pool->pull_pos], pool->table->sizes[pool->pull_pos]);

        /*队列长度减去1。*/
        pool->count -= 1;

        /*滚动游标。*/
        pool->pull_pos = (pool->pull_pos + 1) % pool->table->numbers;

        return 0;
    }

    return -1;
}

int abcdk_pool_push(abcdk_pool_t *pool, const void *buf)
{
    assert(pool != NULL && buf != NULL);

    /*池不能是满的。*/
    if (pool->count < pool->table->numbers)
    {
        /*按游标位置向池子中写入数据。*/
        memcpy(pool->table->pptrs[pool->push_pos], buf, pool->table->sizes[pool->push_pos]);

        /*队列长度加1。*/
        pool->count += 1;

        /*滚动游标。*/
        pool->push_pos = (pool->push_pos + 1) % pool->table->numbers;

        return 0;
    }

    return -1;
}
