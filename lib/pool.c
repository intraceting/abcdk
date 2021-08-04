/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/pool.h"

void abcdk_pool_destroy(abcdk_pool_t *pool)
{
    assert(pool != NULL);

    abcdk_allocator_unref(&pool->table);

    memset(pool, 0, sizeof(*pool));
}

int abcdk_pool_init(abcdk_pool_t *pool, size_t size, size_t number)
{
    assert(pool != NULL && size > 0 && number > 0);

    pool->table = abcdk_allocator_alloc(&size, number, 1);
    if (!pool->table)
        return -1;

    pool->count = 0;
    pool->pull_pos = 0;
    pool->push_pos = 0;

    return 0;
}

ssize_t abcdk_pool_pull(abcdk_pool_t *pool, void *buf, size_t size)
{
    ssize_t len = -1;

    assert(pool != NULL && buf > 0 && size > 0);

    /*池不能是空的。*/
    if (pool->count > 0)
    {
        /*按游标位置从池子中读取数据。*/
        len = ABCDK_MIN(pool->table->sizes[pool->pull_pos], size);
        memcpy(buf, pool->table->pptrs[pool->pull_pos], len);

        /*队列长度减去1。*/
        pool->count -= 1;

        /*滚动游标。*/
        pool->pull_pos = (pool->pull_pos + 1) % pool->table->numbers;
    }

    return len;
}

ssize_t abcdk_pool_push(abcdk_pool_t *pool, const void *buf, size_t size)
{
    ssize_t len = -1;

    assert(pool != NULL && buf > 0 && size > 0);

    /*池不能是满的。*/
    if (pool->count < pool->table->numbers)
    {
        /*按游标位置向池子中写入数据。*/
        len = ABCDK_MIN(pool->table->sizes[pool->push_pos], size);
        memcpy(pool->table->pptrs[pool->push_pos], buf, len);

        /*队列长度加1。*/
        pool->count += 1;

        /*滚动游标。*/
        pool->push_pos = (pool->push_pos + 1) % pool->table->numbers;
    }

    return len;
}
