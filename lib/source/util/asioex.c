/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include "abcdk/util/asioex.h"

/**异步IO对象扩展。*/
struct _abcdk_asioex
{
    /**数组。 */
    abcdk_asio_t **group_ctx;
    
    /**最大数量。*/
    int total_max;

    /**分组数量。*/
    int group_size;

    /**分配游标。 */
    int disp_pos;

};//abcdk_asioex_t

void abcdk_asioex_destroy(abcdk_asioex_t **ctx)
{
    abcdk_asioex_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    for(int i = 0;i<ctx_p->group_size;i++)
        abcdk_asio_destroy(&ctx_p->group_ctx[i]);

    abcdk_heap_free(ctx_p->group_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_asioex_t *abcdk_asioex_create(int group,int max)
{
    abcdk_asioex_t *ctx;

    assert(group > 0 && max >0);

    ctx = (abcdk_asioex_t*)abcdk_heap_alloc(sizeof(abcdk_asioex_t));
    if(!ctx)
        return NULL;

    ctx->group_ctx = (abcdk_asio_t**)abcdk_heap_alloc(sizeof(abcdk_asio_t*)*group);
    if(!ctx->group_ctx)
        goto ERR;

    for (int i = 0; i < group; i++)
    {
        ctx->group_ctx[i] = abcdk_asio_create(abcdk_align(max / group, 2));
        if (!ctx->group_ctx[i])
            goto ERR;
    }

    ctx->group_size = group;
    ctx->total_max = max;
    ctx->disp_pos = 0;

    return ctx;

ERR:

    abcdk_asioex_destroy(&ctx);
    return NULL;
}

void abcdk_asioex_abort(abcdk_asioex_t *ctx)
{
    assert(ctx != NULL);

    for (int i = 0; i < ctx->group_size; i++)
        abcdk_asio_abort(ctx->group_ctx[i]);
}

abcdk_asio_t *abcdk_asioex_dispatch(abcdk_asioex_t *ctx,int idx)
{
    abcdk_asio_t *ctx_ctx_p = NULL;

    assert(ctx != NULL);

    if(idx >= 0)
    {
        if(idx >= ctx->group_size)
            return NULL;

        ctx_ctx_p = ctx->group_ctx[idx];
    }
    else 
    {
        for (int i = 0; i < ctx->group_size; i++)
        {
            if (!ctx_ctx_p)
                ctx_ctx_p = ctx->group_ctx[i];
            else if (abcdk_asio_count(ctx_ctx_p) > abcdk_asio_count(ctx->group_ctx[i]))
                ctx_ctx_p = ctx->group_ctx[i];
        }
    }

    return ctx_ctx_p;
}