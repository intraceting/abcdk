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
    /**配置。*/
    abcdk_iplan_config_t cfg;

    /**存储表(用于查找)。*/
    abcdk_map_t *store_ctx;

    /**监视表(用于遍历)。*/
    abcdk_tree_t *watch_ctx;

    /**同步锁。*/
    abcdk_rwlock_t *locker_ctx;
};//abcdk_iplan_t;

/**节点。 */
typedef struct _abcdk_iplan_node
{
    /*存储标志。0 使用中，1 已删除。*/
    int store_flag;

    /**监视标志。0 未注册，1 已注册。*/
    int watch_flag;

    /**用户环境。*/
    abcdk_context_t *userdata;

}abcdk_iplan_node_t;


static int _abcdk_iplan_sockaddr_len(abcdk_sockaddr_t *addr)
{
    if(addr->family== AF_INET)
        return sizeof(struct sockaddr_in);
    else if(addr->family== AF_INET6)
        return sizeof(struct sockaddr_in6);

    return 0;
}

static void _abcdk_iplan_destructor_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_iplan_t *ctx = (abcdk_iplan_t *)opaque;
    abcdk_sockaddr_t *addr_p = (abcdk_sockaddr_t *)obj->pptrs[ABCDK_MAP_KEY];
    abcdk_iplan_node_t *node_p = (abcdk_iplan_node_t *)obj->pptrs[ABCDK_MAP_VALUE];

    if(!node_p->userdata)
        return;

    if(ctx->cfg.remove_cb)
        ctx->cfg.remove_cb(addr_p,node_p->userdata,ctx->cfg.opaque);

    abcdk_context_unref(&node_p->userdata);
}

static uint64_t _abcdk_iplan_hash_cb(const void* key,size_t size,void *opaque)
{
    abcdk_iplan_t *ctx = (abcdk_iplan_t *)opaque;
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
    abcdk_iplan_t *ctx = (abcdk_iplan_t *)opaque;
    abcdk_sockaddr_t *a = (abcdk_sockaddr_t *)key1;
    abcdk_sockaddr_t *b = (abcdk_sockaddr_t *)key2;

    if (a->family != b->family)
        return -1;

    if (a->family == AF_INET)
    {
        if (a->addr4.sin_addr.s_addr != b->addr4.sin_addr.s_addr)
            return -1;

        if(ctx->cfg.have_port && a->addr4.sin_port != b->addr4.sin_port)
            return -1;
    }
    else if (a->family == AF_INET6)
    {
        if (memcmp(a->addr6.sin6_addr.s6_addr, b->addr6.sin6_addr.s6_addr, 16) != 0)
            return -1;

        if(ctx->cfg.have_port && a->addr6.sin6_port != b->addr6.sin6_port)
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
    abcdk_map_destroy(&ctx_p->store_ctx);
    abcdk_tree_free(&ctx_p->watch_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_iplan_t *abcdk_iplan_create(abcdk_iplan_config_t *cfg)
{
    abcdk_iplan_t *ctx;

    assert(cfg != NULL);

    ctx = (abcdk_iplan_t*)abcdk_heap_alloc(sizeof(abcdk_iplan_t));
    if(!ctx)
        return NULL;

    ctx->cfg = *cfg;

    ctx->store_ctx = abcdk_map_create(100);
    if(!ctx->store_ctx)
        goto ERR;

    ctx->store_ctx->hash_cb = _abcdk_iplan_hash_cb;
    ctx->store_ctx->compare_cb = _abcdk_iplan_compare_cb;
    ctx->store_ctx->destructor_cb = _abcdk_iplan_destructor_cb;
    ctx->store_ctx->opaque = ctx;

    ctx->watch_ctx = abcdk_tree_alloc3(1);
    if(!ctx->watch_ctx)
        goto ERR;

    ctx->locker_ctx = abcdk_rwlock_create();
    if(!ctx->locker_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_iplan_destroy(&ctx);
    return NULL;
}

void abcdk_iplan_remove(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr)
{
    abcdk_object_t *val_p;
    abcdk_iplan_node_t *node_p;
    void *data_p;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    val_p = abcdk_map_find(ctx->store_ctx,addr,_abcdk_iplan_sockaddr_len(addr),0);
    if(!val_p)
        return;

    node_p = (abcdk_iplan_node_t *)val_p->pptrs[ABCDK_MAP_VALUE];

    /*标记已删除。*/
    node_p->store_flag = 1;

    abcdk_map_remove(ctx->store_ctx,addr,_abcdk_iplan_sockaddr_len(addr));

    return;
}

abcdk_context_t *abcdk_iplan_insert(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr,size_t userdata)
{
    abcdk_object_t *val_p;
    abcdk_tree_t *val2_p;
    abcdk_iplan_node_t *node_p;
    void *userdata_p;
    int chk = -1;

    assert(ctx != NULL && addr != NULL && userdata > 0);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    /*优先执行回调函数。*/
    if(ctx->cfg.insert_cb)
    {
        ctx->cfg.insert_cb(addr,&chk,ctx->cfg.opaque);
        if(chk != 0)
            return NULL;
    }

    val_p = abcdk_map_find(ctx->store_ctx, addr,_abcdk_iplan_sockaddr_len(addr),sizeof(abcdk_iplan_node_t));
    if (!val_p)
        return NULL;

    node_p = (abcdk_iplan_node_t *)val_p->pptrs[ABCDK_MAP_VALUE];

    /*如果用户环境未创建，则自动创建。*/
    if (!node_p->userdata && userdata > 0)
        node_p->userdata = abcdk_context_alloc(userdata, NULL);

    /*必须有效。*/
    if(!node_p->userdata)
        goto ERR;

    /*如果未启用监视，则跳过。*/
    if(!ctx->cfg.enable_watch)
        goto END;

    /*如果已经被监视，则跳过。*/
    if(node_p->watch_flag == 1)
        goto END;

    /*引用节点，并创建节点副本。*/
    val2_p = abcdk_tree_alloc(abcdk_object_refer(val_p));
    if(val2_p)
    {   
        /*插入到链表中。*/
        abcdk_tree_insert2(ctx->watch_ctx,val2_p,0);

        /*标记已注册。*/
        node_p->watch_flag = 1;
    }
    else 
    {
        /*创建节点副本失败，反引用。*/
        abcdk_object_unref(&val_p);
        goto ERR;
    }
    
END:    
    
    /*复制用户环境指针。*/
    userdata_p = node_p->userdata;

    return userdata_p;

ERR:

    abcdk_map_remove(ctx->store_ctx,addr,_abcdk_iplan_sockaddr_len(addr));

    return NULL;
}

abcdk_context_t *abcdk_iplan_lookup(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr)
{
    abcdk_object_t *val_p;
    abcdk_iplan_node_t *node_p;
    abcdk_context_t *userdata_p;

    assert(ctx != NULL && addr != NULL);

    val_p = abcdk_map_find(ctx->store_ctx, addr,_abcdk_iplan_sockaddr_len(addr), 0);
    if(!val_p)
        return NULL;

    node_p = (abcdk_iplan_node_t *)val_p->pptrs[ABCDK_MAP_VALUE];

    /*可能已经被删除。*/
    if(node_p->store_flag == 1)
        return NULL;

    userdata_p = node_p->userdata;

    return userdata_p;
}

abcdk_context_t *abcdk_iplan_next(abcdk_iplan_t *ctx,void **it)
{
    abcdk_tree_t *it_p,*it_next_p;
    abcdk_iplan_node_t *node_p;
    abcdk_context_t *userdata_p;

    assert(ctx != NULL && it != NULL);

    it_p = (abcdk_tree_t*)*it;
    *it = NULL;

NEXT:

    if(it_p)
    {
        /*查找下一个节点。*/
        it_next_p = abcdk_tree_sibling(it_p,0);

        /*检查当前节点。*/
        node_p = (abcdk_iplan_node_t *)it_p->obj->pptrs[ABCDK_MAP_VALUE];

        /*可能当前节点已经被删除。*/
        if (node_p->store_flag == 1)
        {
            abcdk_tree_unlink(it_p);
            abcdk_tree_free(&it_p);
        }
    }
    else 
    {
        /*从头开始遍历。*/
        it_next_p = abcdk_tree_child(ctx->watch_ctx,1);
    }

    if(!it_next_p)
        return NULL;

    /*当前节点指向新节点。*/
    it_p = it_next_p;

    /*如果当前节点已经被删除，则遍历下一个。*/
    node_p = (abcdk_iplan_node_t *)it_p->obj->pptrs[ABCDK_MAP_VALUE];
    if(node_p->store_flag == 1)
        goto NEXT;

    /*复制用户指针。*/
    userdata_p = node_p->userdata;

    /*更新迭代器。*/
    *it = (void*)it_p;

    return userdata_p;
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
