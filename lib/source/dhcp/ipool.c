/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/dhcp/ipool.h"

/** IP池。 */
struct _abcdk_ipool
{
    /**地址池。*/
    abcdk_object_t *pool;

    /**启始。 */
    abcdk_sockaddr_t begin;

    /**结束。*/
    abcdk_sockaddr_t end;
    
    /*游标。*/
    uint64_t begin_pos;
    uint64_t end_pos;
    uint64_t pos;

};//abcdk_ipool_t



void abcdk_ipool_destroy(abcdk_ipool_t **ctx)
{
    abcdk_ipool_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->pool);
    abcdk_heap_free(ctx_p);
}

static uint64_t _abcdk_ipool_get_addr_pos(abcdk_sockaddr_t *addr)
{
    uint64_t pos = UINT64_MAX;

    if(addr->family == AF_INET)
    {
        pos = abcdk_endian_b_to_h32(addr->addr4.sin_addr.s_addr);
    }
    else if(addr->family == AF_INET6)
    {
        pos = 0;
        for(int i = 12;i<16;i++)
        {
            pos <<= 8;
            pos += addr->addr6.sin6_addr.__in6_u.__u6_addr8[i];
        }
    }

    return pos;
}

static void _abcdk_ipool_set_addr_pos(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr,uint64_t pos)
{
    addr->family = ctx->begin.family;

    if(addr->family == AF_INET)
    {
        addr->addr4.sin_addr.s_addr = abcdk_endian_h_to_b32(pos);
    }
    else if(addr->family == AF_INET6)
    {
        memcpy(addr->addr6.sin6_addr.__in6_u.__u6_addr8,ctx->begin.addr6.sin6_addr.__in6_u.__u6_addr8,12);
        addr->addr6.sin6_addr.__in6_u.__u6_addr8[12] = (pos >> 24) & 0xff;
        addr->addr6.sin6_addr.__in6_u.__u6_addr8[13] = (pos >> 16) & 0xff;
        addr->addr6.sin6_addr.__in6_u.__u6_addr8[14] = (pos >> 8) & 0xff;
        addr->addr6.sin6_addr.__in6_u.__u6_addr8[15] = pos & 0xff;
    }
}

static  int _abcdk_ipool_init(abcdk_ipool_t *ctx)
{
    uint64_t c = 0;

    if(ctx->begin.family == AF_INET6)
    {
        if(memcmp(ctx->begin.addr6.sin6_addr.__in6_u.__u6_addr8,ctx->end.addr6.sin6_addr.__in6_u.__u6_addr8,12) != 0)
            return -4;
    }

    ctx->begin_pos = _abcdk_ipool_get_addr_pos(&ctx->begin);
    ctx->end_pos = _abcdk_ipool_get_addr_pos(&ctx->end);
    if (ctx->end_pos < ctx->begin_pos)
        return -1;

    c = ctx->end_pos - ctx->begin_pos + 1; // 区间差+1才是数量。

    /*内存有限，限制一下。*/
    if (c <= 0 || c > 0xFFFFFFFFULL)
        return -2;

    ctx->pool = abcdk_object_alloc2(abcdk_align(c,8)/8);
    if(!ctx->pool)
        return -3;

    ctx->pos = ctx->begin_pos;

    return 0;
}

abcdk_ipool_t *abcdk_ipool_create(abcdk_sockaddr_t *begin,abcdk_sockaddr_t *end)
{
    abcdk_ipool_t *ctx;
    int chk;

    assert(begin != NULL && end != NULL);
    assert(begin->family ==  AF_INET || begin->family ==  AF_INET6);
    assert(end->family ==  AF_INET || end->family ==  AF_INET6);
    assert(begin->family == end->family);

    ctx = abcdk_heap_alloc(sizeof(abcdk_ipool_t));
    if(!ctx)
        return NULL;
   
    ctx->begin = *begin;
    ctx->end = *end;

    chk = _abcdk_ipool_init(ctx);
    if(chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_ipool_destroy(&ctx);
    return NULL;
}

abcdk_ipool_t *abcdk_ipool_create2(const char *begin,const char *end)
{
    abcdk_sockaddr_t b,e;
    int chk;

    assert(begin != NULL && end != NULL);

    chk = abcdk_sockaddr_from_string(&b,begin,0);
    if(chk != 0)
        return -22;

    chk = abcdk_sockaddr_from_string(&e,end,0);
    if(chk != 0)
        return -22;

    return abcdk_ipool_create(&b,&e);
}

uint64_t abcdk_ipool_count(abcdk_ipool_t *ctx)
{
    assert(ctx != NULL);

    return ctx->end_pos - ctx->begin_pos + 1;
}

int abcdk_ipool_allocate(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr)
{
    uint64_t c,pos;
    int chk;

    assert(ctx != NULL && addr != NULL);

    c = abcdk_ipool_count(ctx);

    /*限制在一个轮回中查找。*/
    for (uint64_t i = 0; i < c; i++)
    {
        chk = abcdk_bloom_filter(ctx->pool->pptrs[0], ctx->pool->sizes[0], ctx->pos - ctx->begin_pos);

        /*copy.*/
        pos = ctx->pos;

        if(ctx->pos == ctx->end_pos)
            ctx->pos = ctx->begin_pos;
        else
            ctx->pos += 1;

        if (chk != 0)
            continue;

        /*标记占用。*/
        abcdk_bloom_mark(ctx->pool->pptrs[0], ctx->pool->sizes[0], pos - ctx->begin_pos);

        /*填充地址。*/
        _abcdk_ipool_set_addr_pos(ctx,addr, pos);
        return 0;
    }

    return -1;
}

int abcdk_ipool_reclaim(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr)
{
    uint64_t pos; 

    assert(ctx != NULL && addr != NULL);
    assert(addr->family ==  AF_INET || addr->family ==  AF_INET6);

    if(addr->family == AF_INET6)
    {
        if(memcmp(ctx->begin.addr6.sin6_addr.__in6_u.__u6_addr8,addr->addr6.sin6_addr.__in6_u.__u6_addr8,12) != 0)
            return -4;
    }

    pos = _abcdk_ipool_get_addr_pos(addr);

    /*有限的范围。*/
    if (pos <= 0 || pos > 0xFFFFFFFFULL)
        return -2;

    /*不能超过池范围。*/
    if(pos < ctx->begin_pos || pos > ctx->end_pos)
        return -1;

    /*标记空闲。*/
    abcdk_bloom_unset(ctx->pool->pptrs[0], ctx->pool->sizes[0], pos - ctx->begin_pos);

    return 0;
}