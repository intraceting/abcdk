/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/waiter.h"

/** 服务员。*/
struct _abcdk_waiter
{
    /** 
     * 状态。
     * 
     * 1：正常。
     * 2：取消。
    */
    int status;

    /** 锁。*/
    abcdk_mutex_t locker;

    /** 表。*/
    abcdk_map_t map;

};// abcdk_waiter_t;


void abcdk_waiter_free(abcdk_waiter_t **waiter)
{
    abcdk_waiter_t *waiter_p;

    if (!waiter || !*waiter)
        return;

    waiter_p = *waiter;

    abcdk_map_destroy(&waiter_p->map);
    abcdk_mutex_destroy(&waiter_p->locker);

    abcdk_heap_free(waiter_p);

    /*Set NULL(0).*/
    *waiter = NULL;
}

int _abcdk_waiter_compare_cb(const void *key1, size_t size1, const void *key2, size_t size2, void *opaque)
{
    abcdk_waiter_t *waiter = (abcdk_waiter_t *)opaque;

    uint64_t s = ABCDK_PTR2U64(key1, 0);
    uint64_t d = ABCDK_PTR2U64(key2, 0);

    if (s > d)
        return 1;
    if (s < d)
        return -1;

    return 0;
}

void _abcdk_waiter_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_waiter_t *waiter = (abcdk_waiter_t*)opaque;
    abcdk_object_t *obj_p = NULL;

    /*复制应答数据(可能为NULL(0))。*/
    obj_p = (abcdk_object_t *)alloc->pptrs[ABCDK_MAP_VALUE];

    /*解除绑定关系。*/
    alloc->pptrs[ABCDK_MAP_VALUE] = NULL;

    /*可能已经被取走。*/
    if (!obj_p)
        return;

    abcdk_object_unref(&obj_p);
}

abcdk_waiter_t *abcdk_waiter_alloc()
{
    abcdk_waiter_t *waiter;

    waiter = abcdk_heap_alloc(sizeof(abcdk_waiter_t));
    if (!waiter)
        return NULL;

    abcdk_mutex_init2(&waiter->locker, 0);
    abcdk_map_init(&waiter->map, 16);

    waiter->status = 1;
    waiter->map.compare_cb = _abcdk_waiter_compare_cb;
    waiter->map.destructor_cb = _abcdk_waiter_destroy_cb;
    waiter->map.opaque = waiter;    

    return waiter;
}

int abcdk_waiter_register(abcdk_waiter_t *waiter, uint64_t key)
{
    abcdk_object_t *it;
    int chk = -1;

    assert(waiter != NULL);

    abcdk_mutex_lock(&waiter->locker, 1);

    /*申请一个字节的VALUE用于占位。*/
    it = abcdk_map_find(&waiter->map, &key, sizeof(key), 1);
    if (!it)
        goto final;

    /*未修改长度表示重复注册。*/
    if (it->sizes[ABCDK_MAP_VALUE] == -1)
        goto final;

    /*覆盖占位指针和长度。*/
    it->pptrs[ABCDK_MAP_VALUE] = NULL;
    it->sizes[ABCDK_MAP_VALUE] = -1;
    
    chk = 0;

final:

    /*如果有错误，则删除KEY。*/
    if (chk != 0)
        abcdk_map_remove(&waiter->map, &key, sizeof(key));

    abcdk_mutex_unlock(&waiter->locker);

    return chk;
}

uint64_t _abcdk_waiter_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3);
}

abcdk_object_t *abcdk_waiter_wait(abcdk_waiter_t *waiter,uint64_t key, time_t timeout)
{
    time_t time_end;
    time_t time_span;
    abcdk_object_t *obj_p = NULL;
    abcdk_object_t *it;

    assert(waiter != NULL && timeout > 0);

    /*计算过期时间。*/
    time_end = _abcdk_waiter_clock() + timeout;

    abcdk_mutex_lock(&waiter->locker, 1);

    it = abcdk_map_find(&waiter->map, &key, sizeof(key), 0);
    if (!it)
        goto final;

    /*等待到达，或超时。*/
    while (!it->pptrs[ABCDK_MAP_VALUE])
    {
        /*计算剩余超时时长。*/
        time_span = time_end - _abcdk_waiter_clock();
        if (time_span <= 0)
            break;
        
        /*有可能随时被取消。*/
        if (waiter->status != 1)
            break;

        abcdk_mutex_wait(&waiter->locker, time_span);
    }

    /*复制应答数据(可能为NULL(0))。*/
    obj_p = (abcdk_object_t *)it->pptrs[ABCDK_MAP_VALUE];

    /*解除绑定关系。*/
    it->pptrs[ABCDK_MAP_VALUE] = NULL;

    /*删除KEY。*/
    abcdk_map_remove(&waiter->map, &key, sizeof(key));

final:

    abcdk_mutex_unlock(&waiter->locker);

    return obj_p;
}

int abcdk_waiter_response(abcdk_waiter_t *waiter, uint64_t key, abcdk_object_t *obj)
{
    abcdk_object_t *it;
    int chk = -1;

    assert(waiter != NULL && obj != NULL);

    abcdk_mutex_lock(&waiter->locker, 1);

    it = abcdk_map_find(&waiter->map, &key, sizeof(key), 0);
    if (!it)
        goto final;

    /*复制对象指针。*/
    it->pptrs[ABCDK_MAP_VALUE] = (uint8_t*)obj;

    /*通知到达。*/
    abcdk_mutex_signal(&waiter->locker, 1);

    chk = 0;

final:

    abcdk_mutex_unlock(&waiter->locker);

    return chk;
}

void abcdk_waiter_cancel(abcdk_waiter_t *waiter)
{
    assert(waiter != NULL);

    abcdk_mutex_lock(&waiter->locker, 1);
    
    waiter->status = 2;
    abcdk_mutex_signal(&waiter->locker, 1);

    abcdk_mutex_unlock(&waiter->locker);
}

void abcdk_waiter_resume(abcdk_waiter_t *waiter)
{
    assert(waiter != NULL);

    abcdk_mutex_lock(&waiter->locker, 1);
    
    waiter->status = 1;

    abcdk_mutex_unlock(&waiter->locker);
}