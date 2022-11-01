/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/comm/waiter.h"

/** 消息服务员。*/
struct _abcdk_comm_waiter
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

    /** 比较函数。*/
    abcdk_comm_waiter_compare_cb compare_cb;

};// abcdk_comm_waiter_t;


void abcdk_comm_waiter_free(abcdk_comm_waiter_t **waiter)
{
    abcdk_comm_waiter_t *waiter_p;

    if (!waiter || !*waiter)
        return;

    waiter_p = *waiter;

    abcdk_map_destroy(&waiter_p->map);
    abcdk_mutex_destroy(&waiter_p->locker);

    abcdk_heap_free(waiter_p);

    /*Set NULL(0).*/
    *waiter = NULL;
}

int _abcdk_comm_waiter_compare_cb(const void *key1, size_t size1, const void *key2, size_t size2, void *opaque)
{
    abcdk_comm_waiter_t *waiter = (abcdk_comm_waiter_t *)opaque;

    if (waiter->compare_cb)
        return waiter->compare_cb(key1, size1, key2, size2);
    else if (size1 > size2)
        return 1;
    else if (size1 < size2)
        return -1;

    return memcmp(key1, key2, size2);
}

void _abcdk_comm_waiter_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_comm_waiter_t *waiter = (abcdk_comm_waiter_t*)opaque;
    void *val_p = NULL;
    abcdk_comm_queue_t *queue_p = NULL;

    val_p = alloc->pptrs[ABCDK_MAP_VALUE];
    queue_p = ABCDK_PTR2OBJ(abcdk_comm_queue_t *, val_p, 0);

    /*可能已经被取走。*/
    if (!queue_p)
        return;

    abcdk_comm_queue_free(&queue_p);
}

abcdk_comm_waiter_t *abcdk_comm_waiter_alloc()
{
    abcdk_comm_waiter_t *waiter;

    waiter = abcdk_heap_alloc(sizeof(abcdk_comm_waiter_t));
    if (!waiter)
        return NULL;

    abcdk_mutex_init2(&waiter->locker, 0);
    abcdk_map_init(&waiter->map, 16);

    waiter->status = 1;
    waiter->compare_cb = NULL;
    waiter->map.compare_cb = _abcdk_comm_waiter_compare_cb;
    waiter->map.destructor_cb = _abcdk_comm_waiter_destroy_cb;
    waiter->map.opaque = waiter;    

    return waiter;
}

void abcdk_comm_waiter_set_compare_callback(abcdk_comm_waiter_t *waiter,
                                            abcdk_comm_waiter_compare_cb compare_cb)
{
   
    assert(waiter != NULL && compare_cb != NULL);

    abcdk_mutex_lock(&waiter->locker, 1); 

    waiter->compare_cb = compare_cb;

    abcdk_mutex_unlock(&waiter->locker);
}

int abcdk_comm_waiter_request(abcdk_comm_waiter_t *waiter,
                              const void *key, size_t ksize)
{
    void *val_p = NULL;
    abcdk_comm_queue_t *queue = NULL;
    abcdk_object_t *it;
    int chk = -1;

    assert(waiter != NULL && key != NULL && ksize > 0);

    abcdk_mutex_lock(&waiter->locker, 1);


    it = abcdk_map_find(&waiter->map, key, ksize, sizeof(abcdk_comm_queue_t *));
    if (!it)
        goto final;

    queue = abcdk_comm_queue_alloc();
    if (!queue)
        goto final;

    val_p = it->pptrs[ABCDK_MAP_VALUE];

    /*绑定队列指针。*/
    ABCDK_PTR2OBJ(abcdk_comm_queue_t *, val_p, 0) = queue;
    chk = 0;

final:

    /*如果有错误，则删除KEY。*/
    if (chk != 0)
        abcdk_map_remove(&waiter->map, key, ksize);

    abcdk_mutex_unlock(&waiter->locker);

    return chk;
}

abcdk_comm_queue_t *abcdk_comm_waiter_wait(abcdk_comm_waiter_t *waiter,
                                           const void *key, size_t ksize,
                                           size_t max, time_t timeout)
{
    time_t time_end;
    time_t time_span;
    void *val_p = NULL;
    abcdk_comm_queue_t *queue_p = NULL;
    abcdk_object_t *it;

    assert(waiter != NULL && key != NULL && ksize > 0 && max > 0 && timeout > 0);

    /*计算过期时间。*/
    time_end = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3) + timeout;

    abcdk_mutex_lock(&waiter->locker, 1);

    it = abcdk_map_find(&waiter->map, key, ksize, 0);
    if (!it)
        goto final;

    val_p = it->pptrs[ABCDK_MAP_VALUE];

    /*复制队列指针。*/
    queue_p = ABCDK_PTR2OBJ(abcdk_comm_queue_t *, val_p, 0);

    /*等待消息到达，或超时。*/
    while (abcdk_comm_queue_count(queue_p) < max)
    {
        /*计算剩余超时时长。*/
        time_span = time_end - abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3);
        if (time_span <= 0)
            break;
        
        /*有可能随时被取消。*/
        if (waiter->status != 1)
            break;

        abcdk_mutex_wait(&waiter->locker, time_span);
    }

    /*解除绑定关系，删除KEY。*/
    ABCDK_PTR2OBJ(abcdk_comm_queue_t *, val_p, 0) = NULL;
    abcdk_map_remove(&waiter->map, key, ksize);

final:

    abcdk_mutex_unlock(&waiter->locker);

    return queue_p;
}

int abcdk_comm_waiter_response(abcdk_comm_waiter_t *waiter,
                               const void *key, size_t ksize,
                               abcdk_comm_message_t *msg)
{
    void *val_p = NULL;
    abcdk_comm_queue_t *queue_p = NULL;
    abcdk_object_t *it;
    int chk = -1;

    assert(waiter != NULL && key != NULL && ksize > 0 && msg != NULL);

    abcdk_mutex_lock(&waiter->locker, 1);

    it = abcdk_map_find(&waiter->map, key, ksize, 0);
    if (!it)
        goto final;

    val_p = it->pptrs[ABCDK_MAP_VALUE];

    /*复制队列指针。*/
    queue_p = ABCDK_PTR2OBJ(abcdk_comm_queue_t *, val_p, 0);

    chk = abcdk_comm_queue_push(queue_p, msg, 0);
    if (chk != 0)
        goto final;

    /*通知消息到达。*/
    abcdk_mutex_signal(&waiter->locker, 1);

    chk = 0;

final:

    /*如果KEY不在存，则删除消息。*/
    if (chk != 0)
        abcdk_comm_message_unref(&msg);

    abcdk_mutex_unlock(&waiter->locker);

    return chk;
}

void abcdk_comm_waiter_cancel(abcdk_comm_waiter_t *waiter)
{
    assert(waiter != NULL);

    abcdk_mutex_lock(&waiter->locker, 1);
    
    waiter->status = 2;
    abcdk_mutex_signal(&waiter->locker, 1);

    abcdk_mutex_unlock(&waiter->locker);
}

void abcdk_comm_waiter_resume(abcdk_comm_waiter_t *waiter)
{
    assert(waiter != NULL);

    abcdk_mutex_lock(&waiter->locker, 1);
    
    waiter->status = 1;

    abcdk_mutex_unlock(&waiter->locker);
}