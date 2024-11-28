/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/ipool.h"

/** IP池。 */
struct _abcdk_ipool
{
    /**同步锁。*/
    abcdk_rwlock_t *locker_ctx;

    /**地址池。*/
    abcdk_object_t *pool_ctx;

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

}; // abcdk_ipool_t

static uint64_t _abcdk_ipool_get_addr_pos_32b(abcdk_sockaddr_t *addr)
{
    uint64_t pos = UINT64_MAX;

    if (addr->family == AF_INET)
    {
        pos = abcdk_endian_b_to_h32(addr->addr4.sin_addr.s_addr);
    }
    else if (addr->family == AF_INET6)
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

static int _abcdk_ipool_get_addr_pos(abcdk_ipool_t *ctx, abcdk_sockaddr_t *addr, uint64_t *pos)
{
    assert(pos != NULL);

    *pos = UINT64_MAX;

    if (addr->family != ctx->addr_b.family)
        return -1;

    if (addr->family == AF_INET6)
    {
        /*检测前缀是否相同。*/
        if (memcmp(ctx->addr_b.addr6.sin6_addr.s6_addr, addr->addr6.sin6_addr.s6_addr, 12) != 0)
            return -22;
    }

    *pos = _abcdk_ipool_get_addr_pos_32b(addr);

    /*不能超过池范围。*/
    if (*pos < ctx->pool_b || *pos > ctx->pool_e)
        return -14;

    return 0;
}

static void _abcdk_ipool_set_addr_pos(abcdk_ipool_t *ctx, abcdk_sockaddr_t *addr, uint64_t pos)
{
    addr->family = ctx->addr_b.family;

    if (addr->family == AF_INET)
    {
        addr->addr4.sin_addr.s_addr = abcdk_endian_h_to_b32(pos);
    }
    else if (addr->family == AF_INET6)
    {
        memcpy(addr->addr6.sin6_addr.s6_addr, ctx->addr_b.addr6.sin6_addr.s6_addr, 12);
        addr->addr6.sin6_addr.s6_addr[12] = (pos >> 24) & 0xff;
        addr->addr6.sin6_addr.s6_addr[13] = (pos >> 16) & 0xff;
        addr->addr6.sin6_addr.s6_addr[14] = (pos >> 8) & 0xff;
        addr->addr6.sin6_addr.s6_addr[15] = pos & 0xff;
    }
}

void abcdk_ipool_destroy(abcdk_ipool_t **ctx)
{
    abcdk_ipool_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_rwlock_destroy(&ctx_p->locker_ctx);
    abcdk_object_unref(&ctx_p->pool_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_ipool_t *abcdk_ipool_create()
{
    abcdk_ipool_t *ctx;
    int chk;

    ctx = abcdk_heap_alloc(sizeof(abcdk_ipool_t));
    if (!ctx)
        return NULL;

    ctx->locker_ctx = abcdk_rwlock_create();
    if (!ctx->locker_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_ipool_destroy(&ctx);
    return NULL;
}

static int _abcdk_ipool_reset_base(abcdk_ipool_t *ctx, abcdk_sockaddr_t *begin, abcdk_sockaddr_t *end)
{
    uint64_t b, e;
    uint64_t c = 0;
    abcdk_object_t *pool_ctx_p;

    if (begin->family == AF_INET6)
    {
        /*IPV6检测前缀是否相同。*/
        if (memcmp(begin->addr6.sin6_addr.s6_addr, end->addr6.sin6_addr.s6_addr, 12) != 0)
            return -4;
    }

    b = _abcdk_ipool_get_addr_pos_32b(begin);
    e = _abcdk_ipool_get_addr_pos_32b(end);
    if (e < b)
        return -1;

    c = e - b + 1; // 区间差+1才是数量。

    pool_ctx_p = abcdk_object_alloc2(abcdk_align(c, 8) / 8);
    if (!pool_ctx_p)
        return -3;

    abcdk_object_unref(&ctx->pool_ctx);
    ctx->pool_ctx = pool_ctx_p;

    ctx->addr_b = *begin;
    ctx->addr_e = *end;
    ctx->pool_b = b;
    ctx->pool_e = e;

    /*默认关闭。*/
    ctx->dhcp_enable = 0;

    return 0;
}

static int _abcdk_ipool_reset_dhcp(abcdk_ipool_t *ctx, abcdk_sockaddr_t *begin, abcdk_sockaddr_t *end)
{
    uint64_t b, e;
    int chk;

    if (begin->family == AF_INET6)
    {
        /*检测前缀是否相同。*/
        if (memcmp(begin->addr6.sin6_addr.s6_addr, end->addr6.sin6_addr.s6_addr, 12) != 0)
            return -22;
    }

    chk = _abcdk_ipool_get_addr_pos(ctx, begin, &b);
    if (chk != 0)
        return -22;

    chk = _abcdk_ipool_get_addr_pos(ctx, end, &e);
    if (chk != 0)
        return -22;

    if (e < b)
        return -22;

    if (b < ctx->pool_b || e > ctx->pool_e)
        return -1;

    ctx->dhcp_b = b;
    ctx->dhcp_e = e;
    ctx->dhcp_pos = b;
    ctx->dhcp_enable = 1;

    return 0;
}

int abcdk_ipool_reset(abcdk_ipool_t *ctx, abcdk_sockaddr_t *begin, abcdk_sockaddr_t *end,
                      abcdk_sockaddr_t *dhcp_begin, abcdk_sockaddr_t *dhcp_end)
{
    int chk;

    assert(ctx != NULL && begin != NULL && end != NULL);
    assert(begin->family == AF_INET || begin->family == AF_INET6);
    assert(end->family == AF_INET || end->family == AF_INET6);
    assert(begin->family == end->family);

    chk = _abcdk_ipool_reset_base(ctx, begin, end);
    if (chk != 0)
        return -1;

    if (!dhcp_begin || !dhcp_end)
        return 0;

    assert(dhcp_begin->family == AF_INET || dhcp_begin->family == AF_INET6);
    assert(dhcp_end->family == AF_INET || dhcp_end->family == AF_INET6);
    assert(dhcp_begin->family == dhcp_end->family);
    assert(ctx->addr_b.family == dhcp_begin->family);

    chk = _abcdk_ipool_reset_dhcp(ctx, dhcp_begin, dhcp_end);
    if (chk != 0)
        return -2;

    return 0;
}

int abcdk_ipool_reset2(abcdk_ipool_t *ctx, const char *begin, const char *end,
                       const char *dhcp_begin, const char *dhcp_end)
{
    abcdk_sockaddr_t begin_addr = {0}, end_addr = {0};
    abcdk_sockaddr_t dhcp_begin_addr = {0}, dhcp_end_addr = {0};
    int chk;

    assert(ctx != NULL && begin != NULL && end != NULL);

    chk = abcdk_sockaddr_from_string(&begin_addr, begin, 0);
    if (chk != 0)
        return -1;

    chk = abcdk_sockaddr_from_string(&end_addr, end, 0);
    if (chk != 0)
        return -1;

    if (dhcp_begin)
    {
        chk = abcdk_sockaddr_from_string(&dhcp_begin_addr, dhcp_begin, 0);
        if (chk != 0)
            return -1;
    }

    if (dhcp_end)
    {
        chk = abcdk_sockaddr_from_string(&dhcp_end_addr, dhcp_end, 0);
        if (chk != 0)
            return -1;
    }

    if (dhcp_begin && dhcp_end)
        chk = abcdk_ipool_reset(ctx, &begin_addr, &end_addr, &dhcp_begin_addr, &dhcp_end_addr);
    else
        chk = abcdk_ipool_reset(ctx, &begin_addr, &end_addr, NULL, NULL);

    return chk;
}

uint64_t abcdk_ipool_count(abcdk_ipool_t *ctx, int flag)
{
    uint64_t pool_c, dhcp_c;

    assert(ctx != NULL);

    if (ctx->pool_e <= 0 && ctx->pool_b <= 0)
        return 0;

    pool_c = ctx->pool_e - ctx->pool_b + 1;
    dhcp_c = (ctx->dhcp_enable ? (ctx->dhcp_e - ctx->dhcp_b + 1) : 0);

    if (flag == 0)
        return pool_c;
    else if (flag == 1)
        return pool_c - dhcp_c;
    else if (flag == 2)
        return dhcp_c;

    return 0;
}

uint8_t abcdk_ipool_prefix(abcdk_ipool_t *ctx)
{
    uint8_t suffix = 0;
    uint64_t c;

    assert(ctx != NULL);

    c = abcdk_ipool_count(ctx, 0);

    /*计算后缀长度。*/
    while (c > 0)
    {
        suffix += 1;
        c /= 2;
    }

    if (ctx->addr_b.family == AF_INET)
        return 32 - suffix;
    else if (ctx->addr_b.family == AF_INET6)
        return 128 - suffix;

    return 0;
}

int abcdk_ipool_static_request(abcdk_ipool_t *ctx, abcdk_sockaddr_t *addr)
{
    uint64_t pos;
    int chk;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    chk = _abcdk_ipool_get_addr_pos(ctx, addr, &pos);
    if (chk != 0)
        return -22;

    /*标记占用。*/
    chk = abcdk_bloom_mark(ctx->pool_ctx->pptrs[0], ctx->pool_ctx->sizes[0], pos - ctx->pool_b);
    if (chk == 1)
        return -1; // 已经被占用。

    return 0;
}

int abcdk_ipool_static_request2(abcdk_ipool_t *ctx, const char *addr)
{
    abcdk_sockaddr_t s;
    int chk;

    assert(ctx != NULL && addr != NULL);

    chk = abcdk_sockaddr_from_string(&s, addr, 0);
    if (chk != 0)
        return -1;

    return abcdk_ipool_static_request(ctx, &s);
}

int abcdk_ipool_dhcp_request(abcdk_ipool_t *ctx, abcdk_sockaddr_t *addr)
{
    uint64_t c, pos;
    int chk;

    assert(ctx != NULL && addr != NULL);

    /*可能未启用。*/
    if (!ctx->dhcp_enable)
        return -1;

    c = abcdk_ipool_count(ctx, 2);

    /*限制在一个轮回中查找。*/
    for (uint64_t i = 0; i < c; i++)
    {
        chk = abcdk_bloom_filter(ctx->pool_ctx->pptrs[0], ctx->pool_ctx->sizes[0], ctx->dhcp_pos - ctx->pool_b);

        /*copy.*/
        pos = ctx->dhcp_pos;

        if (ctx->dhcp_pos == ctx->dhcp_e)
            ctx->dhcp_pos = ctx->dhcp_b;
        else
            ctx->dhcp_pos += 1;

        if (chk != 0)
            continue;

        /*标记占用。*/
        abcdk_bloom_mark(ctx->pool_ctx->pptrs[0], ctx->pool_ctx->sizes[0], pos - ctx->pool_b);

        /*填充地址。*/
        _abcdk_ipool_set_addr_pos(ctx, addr, pos);
        return 0;
    }

    return -11;
}

int abcdk_ipool_reclaim(abcdk_ipool_t *ctx, abcdk_sockaddr_t *addr)
{
    uint64_t pos;
    int chk;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    chk = _abcdk_ipool_get_addr_pos(ctx, addr, &pos);
    if (chk != 0)
        return -22;

    /*标记空闲。*/
    abcdk_bloom_unset(ctx->pool_ctx->pptrs[0], ctx->pool_ctx->sizes[0], pos - ctx->pool_b);

    return 0;
}

int abcdk_ipool_verify(abcdk_ipool_t *ctx, abcdk_sockaddr_t *addr)
{
    uint64_t pos;
    int chk;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    chk = _abcdk_ipool_get_addr_pos(ctx, addr, &pos);
    if (chk != 0)
        return -22;

    return 0;
}

void abcdk_ipool_rdlock(abcdk_ipool_t *ctx)
{
    assert(ctx != NULL);

    abcdk_rwlock_rdlock(ctx->locker_ctx, 1);
}

void abcdk_ipool_wrlock(abcdk_ipool_t *ctx)
{
    assert(ctx != NULL);

    abcdk_rwlock_wrlock(ctx->locker_ctx, 1);
}

int abcdk_ipool_unlock(abcdk_ipool_t *ctx, int exitcode)
{
    assert(ctx != NULL);

    abcdk_rwlock_unlock(ctx->locker_ctx);

    return exitcode;
}
