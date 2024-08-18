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

    /*同步锁。*/
    abcdk_mutex_t *locker_ctx;

};//abcdk_iplan_t;


static uint8_t *_abcdk_iplan_key_len(abcdk_sockaddr_t *addr)
{
    if(addr->family== AF_INET)
        return sizeof(struct sockaddr_in);
    else if(addr->family== AF_INET6)
        return sizeof(struct sockaddr_in6);

    return 0;
}

static int _abcdk_iplan_compare_cb(const void *key1, size_t size1, const void *key2, size_t size2, void *opaque)
{
    abcdk_sockaddr_t *a = (abcdk_sockaddr_t *)key1;
    abcdk_sockaddr_t *b = (abcdk_sockaddr_t *)key2;

    if (a->family != b->family)
        return -1;

    if (a->family == AF_INET)
    {
        if (a->addr4.sin_addr.s_addr == b->addr4.sin_addr.s_addr)
            return 0;
    }
    else if (a->family == AF_INET6)
    {
        if (memcmp(a->addr6.sin6_addr.__in6_u.__u6_addr8, b->addr6.sin6_addr.__in6_u.__u6_addr8, 16) == 0)
            return 0;
    }

    return 1;
}

void abcdk_iplan_destroy(abcdk_iplan_t **ctx)
{
    abcdk_iplan_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_map_destroy(&ctx_p->table_ctx);
    abcdk_mutex_destroy(&ctx_p->locker_ctx);
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

    ctx->locker_ctx = abcdk_mutex_create();
    if(!ctx->locker_ctx)
        goto ERR;

    ctx->table_ctx->compare_cb = _abcdk_iplan_compare_cb;
    ctx->table_ctx->opaque = ctx;

    return ctx;

ERR:

    abcdk_iplan_destroy(&ctx);
    return NULL;
}

void *abcdk_iplan_remove(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr)
{
    abcdk_object_t *val_p;
    void *data_p;

    assert(ctx != NULL && addr != NULL);

    abcdk_mutex_lock(ctx->locker_ctx,1);

    val_p = abcdk_map_find(ctx,addr,_abcdk_iplan_key_len(addr),0);
    if(!val_p)
        return NULL;

    /*copy.*/
    data_p = val_p->pptrs[ABCDK_MAP_VALUE];

    abcdk_map_remove(ctx,addr,_abcdk_iplan_key_len(addr));

    abcdk_mutex_unlock(ctx->locker_ctx);

    return data_p;
}

int abcdk_iplan_insert(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr,void *data)
{
    abcdk_object_t *val_p;
    int chk = -1;

    assert(ctx != NULL && addr != NULL && data != NULL);

    abcdk_mutex_lock(ctx->locker_ctx,1);

    val_p = abcdk_map_find(ctx,addr,_abcdk_iplan_key_len(addr),1);
    if(!val_p)
        return NULL;

    /*对于已存的不能覆盖，否则会导到应用层错误或内存泄漏。*/
    if(val_p->sizes[ABCDK_MAP_VALUE])
    {
        /*copy.*/
        val_p->pptrs[ABCDK_MAP_VALUE] = data;
        val_p->sizes[ABCDK_MAP_VALUE] = 0;//set 0.
        chk = 0;
    }

    abcdk_mutex_unlock(ctx->locker_ctx);

    return chk;
}

void *abcdk_iplan_lookup(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr)
{
    abcdk_object_t *val_p;
    void *data_p;

    assert(ctx != NULL && addr != NULL);

    abcdk_mutex_lock(ctx->locker_ctx,1);

    val_p = abcdk_map_find(ctx,addr,_abcdk_iplan_key_len(addr),0);
    if(!val_p)
        return NULL;

    /*copy.*/
    data_p = val_p->pptrs[ABCDK_MAP_VALUE];

    abcdk_mutex_unlock(ctx->locker_ctx);

    return data_p;
}