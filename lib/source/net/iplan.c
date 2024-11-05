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
    /**匹配标志。0 仅地址，1 地址和端口。*/
    int compare_flag;

    /**路由表(用于查找)。*/
    abcdk_map_t *table_ctx;

    /**路由表(用于遍历)。*/
    abcdk_tree_t *list_ctx;

    /**同步锁。*/
    abcdk_rwlock_t *locker_ctx;
};//abcdk_iplan_t;

/**IP路径节点。 */
typedef struct _abcdk_iplan_node
{
    /**标志。0 未注册，1 已注册，2 已删除。*/
    int flag;

    /**用户指针。*/
    void *userdata;

}abcdk_iplan_node_t;

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

        if(ctx->compare_flag == 1 && a->addr4.sin_port != b->addr4.sin_port)
            return -1;
    }
    else if (a->family == AF_INET6)
    {
        if (memcmp(a->addr6.sin6_addr.s6_addr, b->addr6.sin6_addr.s6_addr, 16) != 0)
            return -1;

        if(ctx->compare_flag == 1 && a->addr6.sin6_port != b->addr6.sin6_port)
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
    abcdk_tree_free(&ctx_p->list_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_iplan_t *abcdk_iplan_create(int have_port)
{
    abcdk_iplan_t *ctx;

    ctx = (abcdk_iplan_t*)abcdk_heap_alloc(sizeof(abcdk_iplan_t));
    if(!ctx)
        return NULL;

    /*设置匹配标志。*/
    ctx->compare_flag = (have_port ? 1 : 0);

    ctx->table_ctx = abcdk_map_create(100);
    if(!ctx->table_ctx)
        goto ERR;

    ctx->table_ctx->hash_cb = _abcdk_iplan_hash_cb;
    ctx->table_ctx->compare_cb = _abcdk_iplan_compare_cb;
    ctx->table_ctx->opaque = ctx;

    ctx->list_ctx = abcdk_tree_alloc3(1);
    if(!ctx->list_ctx)
        goto ERR;

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
    abcdk_object_t *val_p;
    abcdk_iplan_node_t *node_p;
    void *data_p;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    val_p = abcdk_map_find(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr),0);
    if(!val_p)
        return NULL;

    node_p = (abcdk_iplan_node_t *)val_p->pptrs[ABCDK_MAP_VALUE];

    /*复制用户指针，并标记已删除.*/
    data_p = node_p->userdata;
    node_p->userdata = NULL;
    node_p->flag = 2;

    abcdk_map_remove(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr));

    return data_p;
}

int abcdk_iplan_insert(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr,void *data)
{
    abcdk_object_t *val_p;
    abcdk_iplan_node_t *node_p;
    abcdk_tree_t *val2_p;
    int chk = -1;

    assert(ctx != NULL && addr != NULL && data != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    val_p = abcdk_map_find(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr),sizeof(abcdk_iplan_node_t));
    if(!val_p)
        return -1;

    node_p = (abcdk_iplan_node_t *)val_p->pptrs[ABCDK_MAP_VALUE];

    /*对于已存在的不能覆盖，否则会导到应用层错误或内存泄漏。*/
    if(node_p->flag != 0)
        return -1;

    /*绑定用户指针，并标记已注册。*/
    node_p->userdata = data;
    node_p->flag = 1;

    /*引用节点，并插入到链表中。*/
    val2_p = abcdk_tree_alloc(abcdk_object_refer(val_p));
    if(val2_p)
    {
        abcdk_tree_insert2(ctx->list_ctx,val2_p,0);
        return 0;
    }
    else 
    {
        /*插入失败，反引用，并删除节点。*/
        abcdk_object_unref(&val_p);
        abcdk_map_remove(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr));
        return -1;
    }
}

void *abcdk_iplan_lookup(abcdk_iplan_t *ctx,abcdk_sockaddr_t *addr)
{
    abcdk_object_t *val_p;
    abcdk_iplan_node_t *node_p;
    void *data_p;

    assert(ctx != NULL && addr != NULL);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    val_p = abcdk_map_find(ctx->table_ctx,addr,_abcdk_iplan_key_len(addr),0);
    if(!val_p)
        return NULL;

    node_p = (abcdk_iplan_node_t *)val_p->pptrs[ABCDK_MAP_VALUE];

    /*复制用户指针。*/
    data_p = node_p->userdata;

    return data_p;
}

void *abcdk_iplan_next(abcdk_iplan_t *ctx,abcdk_iplan_iterator_t **it)
{
    abcdk_tree_t *it_p,*it_next_p;
    abcdk_iplan_node_t *node_p;
    void *userdata_p;

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
        if (node_p->flag == 2)
        {
            abcdk_tree_unlink(it_p);
            abcdk_tree_free(&it_p);
        }
    }
    else 
    {
        /*从头开始遍历。*/
        it_next_p = abcdk_tree_child(ctx->list_ctx,1);
    }

    if(!it_next_p)
        return NULL;

    /*当前节点指向新节点。*/
    it_p = it_next_p;

    /*如果当前节点已经被删除，则遍历下一个。*/
    node_p = (abcdk_iplan_node_t *)it_p->obj->pptrs[ABCDK_MAP_VALUE];
    if(node_p->flag == 2)
        goto NEXT;

    /*复制用户指针。*/
    userdata_p = node_p->userdata;

    /*更新迭代器。*/
    *it = (abcdk_iplan_iterator_t*)it_p;

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
