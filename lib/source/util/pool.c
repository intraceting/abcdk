/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#include "abcdk/util/pool.h"

/**
 * 一个简单的池子。
*/
struct _abcdk_pool
{
    /**
     * 池子。
     * 
     * @note 尽量不要直接修改。
    */
    abcdk_object_t *table;

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

} ;//abcdk_pool_t;


void abcdk_pool_destroy(abcdk_pool_t **ctx)
{
    abcdk_pool_t *ctx_p = NULL;

    if(!ctx||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->table);
    abcdk_heap_free(ctx_p);
}

static int _abcdk_pool_init(abcdk_pool_t *pool, size_t size, size_t number)
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

abcdk_pool_t *abcdk_pool_create(size_t size, size_t number)
{
    abcdk_pool_t *ctx;
    int chk;

    ctx = (abcdk_pool_t*)abcdk_heap_alloc(sizeof(abcdk_pool_t));
    if(!ctx)
        return NULL;

    chk = _abcdk_pool_init(ctx,size,number);
    if(chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_pool_destroy(&ctx);

    return NULL;
}

int abcdk_pool_pull(abcdk_pool_t *ctx, void *buf)
{
    assert(ctx != NULL && buf != NULL);

    /*池不能是空的。*/
    if (ctx->count > 0)
    {
        /*按游标位置从池子中读取数据。*/
        memcpy(buf, ctx->table->pptrs[ctx->pull_pos], ctx->table->sizes[ctx->pull_pos]);

        /*队列长度减去1。*/
        ctx->count -= 1;

        /*滚动游标。*/
        ctx->pull_pos = (ctx->pull_pos + 1) % ctx->table->numbers;

        return 0;
    }

    return -1;
}

int abcdk_pool_push(abcdk_pool_t *ctx, const void *buf)
{
    assert(ctx != NULL && buf != NULL);

    /*池不能是满的。*/
    if (ctx->count < ctx->table->numbers)
    {
        /*按游标位置向池子中写入数据。*/
        memcpy(ctx->table->pptrs[ctx->push_pos], buf, ctx->table->sizes[ctx->push_pos]);

        /*队列长度加1。*/
        ctx->count += 1;

        /*滚动游标。*/
        ctx->push_pos = (ctx->push_pos + 1) % ctx->table->numbers;

        return 0;
    }

    return -1;
}

size_t abcdk_pool_count(abcdk_pool_t *ctx)
{
    assert(ctx != NULL);

    return ctx->count;
}

