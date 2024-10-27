/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/net/iplan.h"

/**IP路径。 */
struct _abcdk_iplan
{
    /*路由表。*/
    abcdk_map_t *table_ctx;

    /**同步锁。*/
    abcdk_rwlock_t *locker_ctx;
};//abcdk_iplan_t;

static int _abcdk_iplan_key_len(abcdk_sockaddr_t *addr)
{
    if(addr->family== AF_INET)
        return sizeof(struct sockaddr_in);
    else if(addr->family== AF_INET6)
        return sizeof(struct sockaddr_in6);

    return 0;
}

static uint64_t _abcdk_iplan_hash_cb(const void* key,size_t size,void *opaque)
{
    uint64_t hs = UINT64_MAX;

    abcdk_sockaddr_t *a = (abcdk_sockaddr_t *)key;

    if (a->family == AF_INET)
    {
        hs = abcdk_hash_bkdr64(&a->addr4.sin_addr.s_addr,4);
    }
    else if (a->family == AF_INET6)
    {
        hs = abcdk_hash_bkdr64(a->addr6.sin6_addr.s6_addr,16);
    }

    return hs;
}

static int _abcdk_iplan_compare_cb(const void *key1, size_t size1, const void *key2, size_t size2, void *opaque)
{
    abcdk_sockaddr_t *a = (abcdk_sockaddr_t *)key1;
    abcdk_sockaddr_t *b = (abcdk_sockaddr_t *)key2;

    if (a->family != b->family)
        return -1;

    if (a->family == AF_INET)
    {
        if (a->addr4.sin_addr.s_addr != b->addr4.sin_addr.s_addr)
            return -1;
    }
    else if (a->family == AF_INET6)
    {
        if (memcmp(a->addr6.sin6_addr.s6_addr, b->addr6.sin6_addr.s6_addr, 16) != 0)
            return -1;
    }

    return 0;
}

void abcdk_iplan_destroy(abcdk_iplan_t **ctx)
{
    abcdk_iplan_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_rwlock_destroy(&ctx_p->locker_ctx);
    abcdk_map_destroy(&ctx_p->table_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_iplan_t *abcdk_iplan_create()
{
    abcdk_iplan_t *ctx;

    ctx = (abcdk_iplan_t*)abcdk_heap_alloc(sizeof(abcdk_iplan_t));
    if(!ctx)
        return NULL;

    ctx->table_ctx = abcdk_map_create(10);
    if(!ctx->table_ctx)
        goto ERR;

    ctx->table_ctx->hash_cb = _abcdk_iplan_hash_cb;
    ctx->table_ctx->compare_cb = _abcdk_iplan_compare_cb;
    ctx->table_ctx->opaque = ctx;

    ctx->locker_ctx = abcdk_rwlock_create();
    if(!ctx->locker_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_iplan_destroy(&ctx);
    return NULL;
}

void *abcdk_iplan_remove(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr)
{
    //char addrstr[100] = {0};
    abcdk_object_t *val_p;
    void *data_p;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    val_p = abcdk_map_find(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr),0);
    if(!val_p)
        return NULL;

    /*copy.*/
    data_p = val_p->pptrs[ABCDK_MAP_VALUE];

    abcdk_map_remove(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr));

    //abcdk_sockaddr_to_string(addrstr, addr);
    //abcdk_trace_output(LOG_DEBUG,"从路由表中删除地址(%s)",addrstr);

    return data_p;
}

int abcdk_iplan_insert(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr,void *data)
{
    //char addrstr[100] = {0};
    abcdk_object_t *val_p;
    int chk = -1;

    assert(ctx != NULL && addr != NULL && data != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    val_p = abcdk_map_find(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr),1);
    if(!val_p)
        return -1;

    /*对于已存的不能覆盖，否则会导到应用层错误或内存泄漏。*/
    if(val_p->sizes[ABCDK_MAP_VALUE] == 0)
        return -1;
    
    /*copy.*/
    val_p->pptrs[ABCDK_MAP_VALUE] = data;
    val_p->sizes[ABCDK_MAP_VALUE] = 0;//set 0.

    //abcdk_sockaddr_to_string(addrstr, addr);
    //abcdk_trace_output(LOG_DEBUG,"向路由表中添加地址(%s)",addrstr);

    return 0;
}

void *abcdk_iplan_lookup(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr)
{
    //char addrstr[100] = {0};
    abcdk_object_t *val_p;
    void *data_p;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    //abcdk_sockaddr_to_string(addrstr, addr);
    //abcdk_trace_output(LOG_DEBUG,"在路由表中查找地址(%s)。",addrstr);

    val_p = abcdk_map_find(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr),0);
    if(!val_p)
        return NULL;

    /*copy.*/
    data_p = val_p->pptrs[ABCDK_MAP_VALUE];

    //abcdk_sockaddr_to_string(addrstr, addr);
    //abcdk_trace_output(LOG_DEBUG,"在路由表中查找地址(%s)成功。",addrstr);

    return data_p;
}

void abcdk_iplan_rdlock(abcdk_iplan_t *ctx)
{
    assert(ctx != NULL);

    abcdk_rwlock_rdlock(ctx->locker_ctx,1);
}


void abcdk_iplan_wrlock(abcdk_iplan_t *ctx)
{
    assert(ctx != NULL);

    abcdk_rwlock_wrlock(ctx->locker_ctx,1);
}


int abcdk_iplan_unlock(abcdk_iplan_t *ctx,int exitcode)
{
    assert(ctx != NULL);

    abcdk_rwlock_unlock(ctx->locker_ctx);

    return exitcode;
}
