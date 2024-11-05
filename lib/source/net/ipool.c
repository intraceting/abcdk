/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/net/ipool.h"

/** IP池。 */
struct _abcdk_ipool
{
    /**地址池。*/
    abcdk_object_t *pool_ctx;

    /**同步锁。*/
    abcdk_spinlock_t *locker_ctx;


    /**地址的起止。 */
    abcdk_sockaddr_t addr_b;
    abcdk_sockaddr_t addr_e;

    /**池的范围。 */
    uint64_t pool_b;
    uint64_t pool_e;

    /**HDCP状态. 0 停用，!0 启用。 */
    int dhcp_enable;
    
    /**DHCP的范围。*/
    uint64_t dhcp_b;
    uint64_t dhcp_e;

    /**DHCP游标。*/
    uint64_t dhcp_pos;

};//abcdk_ipool_t



void abcdk_ipool_destroy(abcdk_ipool_t **ctx)
{
    abcdk_ipool_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_spinlock_destroy(&ctx_p->locker_ctx);
    abcdk_object_unref(&ctx_p->pool_ctx);
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
        for (int i = 12; i < 16; i++)
        {
            pos <<= 8;
            pos += addr->addr6.sin6_addr.s6_addr[i];
        }
    }

    return pos;
}

static void _abcdk_ipool_set_addr_pos(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr,uint64_t pos)
{
    addr->family = ctx->addr_b.family;

    if(addr->family == AF_INET)
    {
        addr->addr4.sin_addr.s_addr = abcdk_endian_h_to_b32(pos);
    }
    else if(addr->family == AF_INET6)
    {
        memcpy(addr->addr6.sin6_addr.s6_addr,ctx->addr_b.addr6.sin6_addr.s6_addr,12);
        addr->addr6.sin6_addr.s6_addr[12] = (pos >> 24) & 0xff;
        addr->addr6.sin6_addr.s6_addr[13] = (pos >> 16) & 0xff;
        addr->addr6.sin6_addr.s6_addr[14] = (pos >> 8) & 0xff;
        addr->addr6.sin6_addr.s6_addr[15] = pos & 0xff;
    }
}

static int _abcdk_ipool_init(abcdk_ipool_t *ctx)
{
    uint64_t c = 0;

    ctx->locker_ctx = abcdk_spinlock_create();
    if(!ctx->locker_ctx)
        return -1;

    if(ctx->addr_b.family == AF_INET6)
    {
        /*检测前缀是否相同。*/
        if(memcmp(ctx->addr_b.addr6.sin6_addr.s6_addr,ctx->addr_e.addr6.sin6_addr.s6_addr,12) != 0)
            return -4;
    }

    ctx->pool_b = _abcdk_ipool_get_addr_pos(&ctx->addr_b);
    ctx->pool_e = _abcdk_ipool_get_addr_pos(&ctx->addr_e);
    if (ctx->pool_e < ctx->pool_b)
        return -1;

    c = ctx->pool_e - ctx->pool_b + 1; // 区间差+1才是数量。

    /*内存有限，限制一下。*/
    if (c <= 0 || c > 0xFFFFFFFFULL)
        return -2;

    ctx->pool_ctx = abcdk_object_alloc2(abcdk_align(c,8)/8);
    if(!ctx->pool_ctx)
        return -3;

    /*默认关闭。*/
    ctx->dhcp_enable = 0;

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
   
    ctx->addr_b = *begin;
    ctx->addr_e = *end;

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
        return NULL;

    chk = abcdk_sockaddr_from_string(&e,end,0);
    if(chk != 0)
        return NULL;

    return abcdk_ipool_create(&b,&e);
}

abcdk_ipool_t *abcdk_ipool_create3(const char *net,int prefix)
{
    abcdk_sockaddr_t n,b,e;
    int chk;

    assert(net != NULL && prefix >= 0);

    chk = abcdk_sockaddr_from_string(&n,net,0);
    if(chk != 0)
        return NULL;
}

int abcdk_ipool_set_dhcp_range(abcdk_ipool_t *ctx,abcdk_sockaddr_t *begin,abcdk_sockaddr_t *end)
{
    uint64_t b,e;

    assert(ctx != NULL && begin != NULL && end != NULL);
    assert(begin->family ==  AF_INET || begin->family ==  AF_INET6);
    assert(end->family ==  AF_INET || end->family ==  AF_INET6);
    assert(begin->family == end->family);
    assert(ctx->addr_b.family == begin->family);

    if(ctx->addr_b.family == AF_INET6)
    {
        /*检测前缀是否相同。*/
        if(memcmp(begin->addr6.sin6_addr.s6_addr,end->addr6.sin6_addr.s6_addr,12) != 0)
            return -22;

        /*检测前缀是否相同(是否在当前池范围内)。*/
        if(memcmp(ctx->addr_b.addr6.sin6_addr.s6_addr,begin->addr6.sin6_addr.s6_addr,12) != 0)
            return -1;
    }

    b = _abcdk_ipool_get_addr_pos(begin);
    e = _abcdk_ipool_get_addr_pos(end);
    if (e < b)
        return -22;

    if(b < ctx->pool_b || e > ctx->pool_e)
        return -1;

    ctx->dhcp_b = b;
    ctx->dhcp_e = e;
    ctx->dhcp_pos = b;
    ctx->dhcp_enable = 1;

    return 0;
}

int abcdk_ipool_set_dhcp_range2(abcdk_ipool_t *ctx,const char *begin,const char *end)
{
    abcdk_sockaddr_t b,e;
    int chk;

    assert(ctx != NULL && begin != NULL && end != NULL);

    chk = abcdk_sockaddr_from_string(&b,begin,0);
    if(chk != 0)
        return -1;

    chk = abcdk_sockaddr_from_string(&e,end,0);
    if(chk != 0)
        return -1;

    return abcdk_ipool_set_dhcp_range(ctx,&b,&e);  
}

uint64_t abcdk_ipool_count(abcdk_ipool_t *ctx,int flag)
{
    uint64_t pool_c,dhcp_c;

    assert(ctx != NULL);

    pool_c = ctx->pool_e - ctx->pool_b +1;
    dhcp_c = (ctx->dhcp_enable?(ctx->dhcp_e - ctx->dhcp_b +1):0);

    if(flag == 0)
        return pool_c;
    else if(flag == 1)
        return pool_c - dhcp_c;
    else if(flag == 2)
        return dhcp_c;

    return 0;
}

uint8_t abcdk_ipool_prefix(abcdk_ipool_t *ctx)
{
    uint8_t suffix = 0;
    uint64_t c;

    assert(ctx != NULL);

    c = abcdk_ipool_count(ctx,0);

    /*计算后缀长度。*/
    while(c >0)
    {
        suffix += 1;
        c /= 2;
    }

    if(ctx->addr_b.family == AF_INET)
        return 32-suffix;
    else if(ctx->addr_b.family == AF_INET6)
        return 128-suffix;
    
    return 0;
}

int abcdk_ipool_static_request(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr)
{
    uint64_t pos;
    int chk;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family ==  AF_INET || addr->family ==  AF_INET6);
    assert(addr->family == ctx->addr_b.family);

    if(addr->family == AF_INET6)
    {
        /*检测前缀是否相同。*/
        if(memcmp(ctx->addr_b.addr6.sin6_addr.s6_addr,addr->addr6.sin6_addr.s6_addr,12) != 0)
            return -4;
    }

    pos = _abcdk_ipool_get_addr_pos(addr);

    /*有限的范围。*/
    if (pos <= 0 || pos > 0xFFFFFFFFULL)
        return -2;

    /*不能超过池范围。*/
    if(pos < ctx->pool_b || pos > ctx->pool_e)
        return -1;

    /*标记占用。*/
    chk = abcdk_bloom_mark(ctx->pool_ctx->pptrs[0], ctx->pool_ctx->sizes[0], pos - ctx->pool_b);
    if(chk == 1)
        return -1;//已经被占用。

    return 0;
}

int abcdk_ipool_static_request2(abcdk_ipool_t *ctx,const char *addr)
{
    abcdk_sockaddr_t s;
    int chk;

    assert(ctx != NULL && addr != NULL);

    chk = abcdk_sockaddr_from_string(&s,addr,0);
    if(chk != 0)
        return -1;

    return abcdk_ipool_static_request(ctx,&s);
}

int abcdk_ipool_dhcp_request(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr)
{
    uint64_t c,pos;
    int chk;

    assert(ctx != NULL && addr != NULL);
    
    /*可能未启用。*/
    if(!ctx->dhcp_enable)
        return -1;

    c = abcdk_ipool_count(ctx,2);

    /*限制在一个轮回中查找。*/
    for (uint64_t i = 0; i < c; i++)
    {
        chk = abcdk_bloom_filter(ctx->pool_ctx->pptrs[0], ctx->pool_ctx->sizes[0], ctx->dhcp_pos - ctx->pool_b);

        /*copy.*/
        pos = ctx->dhcp_pos;

        if(ctx->dhcp_pos == ctx->dhcp_e)
            ctx->dhcp_pos = ctx->dhcp_b;
        else
            ctx->dhcp_pos += 1;

        if (chk != 0)
            continue;

        /*标记占用。*/
        abcdk_bloom_mark(ctx->pool_ctx->pptrs[0], ctx->pool_ctx->sizes[0], pos - ctx->pool_b);

        /*填充地址。*/
        _abcdk_ipool_set_addr_pos(ctx,addr, pos);
        return 0;
    }

    return -11;
}

int abcdk_ipool_reclaim(abcdk_ipool_t *ctx,abcdk_sockaddr_t *addr)
{
    uint64_t pos; 

    assert(ctx != NULL && addr != NULL);
    assert(addr->family ==  AF_INET || addr->family ==  AF_INET6);
    assert(addr->family == ctx->addr_b.family);

    if(addr->family == AF_INET6)
    {
        /*检测前缀是否相同。*/
        if(memcmp(ctx->addr_b.addr6.sin6_addr.s6_addr,addr->addr6.sin6_addr.s6_addr,12) != 0)
            return -4;
    }

    pos = _abcdk_ipool_get_addr_pos(addr);

    /*有限的范围。*/
    if (pos <= 0 || pos > 0xFFFFFFFFULL)
        return -2;

    /*不能超过池范围。*/
    if(pos < ctx->pool_b || pos > ctx->pool_e)
        return -1;

    /*标记空闲。*/
    abcdk_bloom_unset(ctx->pool_ctx->pptrs[0], ctx->pool_ctx->sizes[0], pos - ctx->pool_b);

    return 0;
}

void abcdk_ipool_lock(abcdk_ipool_t *ctx)
{
    assert(ctx != NULL);

    abcdk_spinlock_lock(ctx->locker_ctx,1);
}

int abcdk_ipool_unlock(abcdk_ipool_t *ctx,int exitcode)
{
    assert(ctx != NULL);

    abcdk_spinlock_unlock(ctx->locker_ctx);

    return exitcode;
}
