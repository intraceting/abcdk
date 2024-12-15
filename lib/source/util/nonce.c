/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/nonce.h"

/**简单的NONCE环境。 */
struct _abcdk_nonce
{
    /**同步锁。*/
    abcdk_rwlock_t *locker_ctx;

    /**列表配置。*/
    abcdk_registry_config_t list_cfg;

    /**列表。*/
    abcdk_registry_t *list_ctx; 

    /**看门狗。*/
    abcdk_timer_t *dog_ctx;

    /**时间误差(毫秒)。*/
    uint64_t time_diff;


};// abcdk_nonce_t;

/**节点。*/
typedef struct _abcdk_nonce_node
{
    /*KEY。*/
    uint8_t key[32];
    
    /*计数器。*/
    int count;

}abcdk_nonce_node_t;


void abcdk_nonce_destroy(abcdk_nonce_t **ctx)
{
    abcdk_nonce_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_timer_destroy(&ctx_p->dog_ctx);
    abcdk_rwlock_destroy(&ctx_p->locker_ctx);
    abcdk_registry_destroy(&ctx_p->list_ctx);
    abcdk_heap_free(ctx_p);
}

static uint64_t _abcdk_nonce_node_key_size_cb(const void *key, void *opaque);
static uint64_t _abcdk_nonce_dog_routine_cb(void *opaque);

abcdk_nonce_t *abcdk_nonce_create()
{
    abcdk_nonce_t *ctx;

    ctx = (abcdk_nonce_t *)abcdk_heap_alloc(sizeof(abcdk_nonce_t));
    if(!ctx)
        return NULL;

    ctx->list_cfg.enable_watch = 1;
    ctx->list_cfg.key_size_cb = _abcdk_nonce_node_key_size_cb;
    ctx->list_cfg.opaque = ctx;
    ctx->list_ctx = abcdk_registry_create(&ctx->list_cfg);
    if(!ctx->list_ctx)
        goto ERR;

    ctx->locker_ctx = abcdk_rwlock_create();
    if(!ctx->locker_ctx)
        goto ERR;

    ctx->dog_ctx = abcdk_timer_create(_abcdk_nonce_dog_routine_cb,ctx);
    if (!ctx->dog_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_nonce_destroy(&ctx);
    return NULL;
}

static uint64_t _abcdk_nonce_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_REALTIME, 3);
}

int abcdk_nonce_reset(abcdk_nonce_t *ctx, uint64_t time_diff)
{
    assert(ctx != NULL && time_diff != 0);

    abcdk_rwlock_wrlock(ctx->locker_ctx,1);

    /*复制时间误差。*/
    ctx->time_diff = time_diff;

    abcdk_rwlock_unlock(ctx->locker_ctx);

    return 0;
}

static int _abcdk_nonce_check(abcdk_nonce_t *ctx, const uint8_t key[32])
{
    uint64_t now_tick, old_tick, diff_tick;
    int chk;

    /*获取现在的时间点。*/
    now_tick = _abcdk_nonce_clock();

    /**
     * |RANDOM   |TIME-MS |SEQ-NUM |
     * |---------|--------|--------|
     * |16 bytes |8 bytes |8 bytes |
    */

    /*读取旧的时间点。*/
    old_tick = abcdk_bloom_read_number(key, 32, 16 * 8, 64);

    /*计算误差。*/
    diff_tick = ((now_tick > old_tick) ? (now_tick - old_tick) : (old_tick - now_tick));   

    /*比较误差。*/
    chk = (diff_tick > ctx->time_diff ? -1 : 0);
  
    return chk;
}

static int _abcdk_nonce_generate(abcdk_nonce_t *ctx,uint8_t key[32])
{
    /**
     * |RANDOM   |TIME-MS |SEQ-NUM |
     * |---------|--------|--------|
     * |16 bytes |8 bytes |8 bytes |
    */

    abcdk_rand_bytes(key, 16, 5);
    abcdk_bloom_write_number(key, 48, 16 * 8, 64, _abcdk_nonce_clock());
    abcdk_bloom_write_number(key, 48, 24 * 8, 64, abcdk_sequence_num());

    return 0;
}

int abcdk_nonce_generate(abcdk_nonce_t *ctx,uint8_t key[32])
{
    int chk;

    assert(ctx != NULL && key != NULL);

    chk = _abcdk_nonce_generate(ctx,key);
    if(chk != 0)
        return -1;

    return 0;
}

static uint64_t _abcdk_nonce_node_key_size_cb(const void *key, void *opaque)
{
    return 32;
}

static abcdk_context_t *_abcdk_nonce_node_insert(abcdk_nonce_t *ctx,const uint8_t key[32])
{
    abcdk_context_t *registry_p = NULL;

    abcdk_registry_wrlock(ctx->list_ctx);

    registry_p = abcdk_registry_insert(ctx->list_ctx,key,sizeof(abcdk_nonce_node_t));
    if(registry_p)
        registry_p = abcdk_context_refer(registry_p);/*增加引用计数。*/

    abcdk_registry_unlock(ctx->list_ctx, 0); 

    return registry_p;
}

static void _abcdk_nonce_node_remote(abcdk_nonce_t *ctx,const uint8_t key[32])
{
    abcdk_registry_wrlock(ctx->list_ctx);

    abcdk_registry_remove(ctx->list_ctx,key);

    abcdk_registry_unlock(ctx->list_ctx, 0); 
}

abcdk_context_t *_abcdk_nonce_node_next(abcdk_nonce_t *ctx,void **it)
{
    abcdk_context_t *registry_p = NULL;

    abcdk_registry_rdlock(ctx->list_ctx);

    registry_p = abcdk_registry_next(ctx->list_ctx,it);
    if(registry_p)
        registry_p = abcdk_context_refer(registry_p);/*增加引用计数。*/

    abcdk_registry_unlock(ctx->list_ctx, 0); 

    return registry_p;
}

static int _abcdk_nonce_node_update(abcdk_nonce_t *ctx,const uint8_t key[32])
{
    abcdk_context_t *node_p = NULL;
    abcdk_nonce_node_t *node_ctx_p = NULL;
    int chk = -1;

    node_p = _abcdk_nonce_node_insert(ctx, key);
    if (!node_p)
        return -1;

    node_ctx_p = abcdk_context_get_userdata(node_p);

    abcdk_context_wrlock(node_p);//lock.

    /*累加次数。*/
    node_ctx_p->count += 1;

    /*超过一次表示重复收到。*/
    if (node_ctx_p->count <= 1)
    {
        /*记录KEY。*/
        memcpy(node_ctx_p->key,key,32);
    }
    
    /*复制次数。*/
    chk = node_ctx_p->count;

    abcdk_context_unlock(node_p,0);//unlock.

    abcdk_context_unref(&node_p);//free.

    return chk;
}

static int _abcdk_nonce_dog_next_node(abcdk_nonce_t *ctx,abcdk_nonce_node_t *node,void **it)
{
    abcdk_context_t *node_p = NULL;
    abcdk_nonce_node_t *node_ctx_p = NULL;

    node_p = _abcdk_nonce_node_next(ctx, it);
    if (!node_p)
        return 0;

    node_ctx_p = abcdk_context_get_userdata(node_p);

    abcdk_context_rdlock(node_p);//lock.

    /*复制。*/
    *node = *node_ctx_p;

    abcdk_context_unlock(node_p,0);//unlock.

    abcdk_context_unref(&node_p);//free.

    return 1;
}

static void _abcdk_nonce_dog_process_node(abcdk_nonce_t *ctx, abcdk_nonce_node_t *node)
{
    uint64_t now_tick, old_tick, diff_tick;
    int chk;

    abcdk_rwlock_rdlock(ctx->locker_ctx, 1);
    chk = _abcdk_nonce_check(ctx, node->key);
    abcdk_rwlock_unlock(ctx->locker_ctx);

    /*超过时差范围，删除。*/
    if (chk != 0)
        _abcdk_nonce_node_remote(ctx, node->key);
}

static uint64_t _abcdk_nonce_dog_routine_cb(void *opaque)
{
    abcdk_nonce_t *ctx = (abcdk_nonce_t*)opaque;
    abcdk_nonce_node_t node = {0};
    void *it_p = NULL;
    int chk;

    while(1)
    {
        chk = _abcdk_nonce_dog_next_node(ctx,&node,&it_p);
        if(chk == 0)
            break;

        _abcdk_nonce_dog_process_node(ctx,&node);

        memset(&node,0,sizeof(node));//clear.
    }

    return 1000;
}

int abcdk_nonce_check(abcdk_nonce_t *ctx, const uint8_t key[32])
{
    uint64_t now_tick, old_tick, diff_tick;
    abcdk_context_t *node_p = NULL;
    abcdk_nonce_node_t *node_ctx_p = NULL;
    int chk = -1;

    assert(ctx != NULL && key != NULL);

    abcdk_rwlock_rdlock(ctx->locker_ctx, 1);
    chk = _abcdk_nonce_check(ctx, key);
    abcdk_rwlock_unlock(ctx->locker_ctx);

    /*超过误差范围，直接返回。*/
    if (chk != 0)
        return -1;

    /*更新节点。*/
    chk = _abcdk_nonce_node_update(ctx,key);
    if(chk > 1)
        return -2;

    return 0;
}