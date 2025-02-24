/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/odbc/pool.h"

#if defined(__SQL_H) && defined(__SQLEXT_H)

/**
 * ODBC连接池。
*/
struct _abcdk_odbcpool
{
    /** 魔法数。*/
    uint32_t magic;

    /** 同步锁。*/
    abcdk_mutex_t *mutex;

    /** 连接池。*/
    size_t pool_size;
    abcdk_pool_t *pool;

    /** 退出标志。*/
    int exitflag;

    /** 弹出的连接数量。*/
    volatile size_t pop_nbs;

    /** 连接数据库回调函数指针。*/
    abcdk_odbcpool_connect_cb connect_cb;

    /** 环境指针。*/
    void *opaque;

};//abcdk_odbcpool_t;


time_t _abcdk_odbcpool_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC,3);
}

void abcdk_odbcpool_destroy(abcdk_odbcpool_t **ctx)
{
    abcdk_odbcpool_t *p = NULL;
    abcdk_odbc_t *odbc_p = NULL;

    if (!ctx || !*ctx)
        return;

    p = *ctx;
    *ctx = NULL;

    abcdk_mutex_lock(p->mutex,1);
    
    p->exitflag = 1;

    /*通知所有pop停止等待。*/
    abcdk_mutex_signal(p->mutex,1);

    /*等待所有已弹出连接被回收。*/
    while (p->pop_nbs > 0)
    {
        abcdk_mutex_wait(p->mutex, 60 * 60 * 1000);
        ABCDK_ASSERT(p->pop_nbs <= 0, "当您看见这个消息时，表示已弹出的连接还有未被回收的。");
    }

    /*关闭所有连接。*/
    while (abcdk_pool_pull(p->pool, &odbc_p) == 0)
    {
        ABCDK_ASSERT(odbc_p != NULL, "当您看见这个消息时，表示应用程序已经发生严重的错误。");
        abcdk_odbc_disconnect(odbc_p);
        abcdk_odbc_free(&odbc_p);
    }

    abcdk_mutex_unlock(p->mutex);

    abcdk_pool_destroy(&p->pool);
    abcdk_mutex_destroy(&p->mutex);

    abcdk_heap_free(p);
}

abcdk_odbcpool_t *abcdk_odbcpool_create(size_t size, abcdk_odbcpool_connect_cb connect_cb, void *opaque)
{
    abcdk_odbcpool_t *p = NULL;
    static volatile uint32_t magic = 1;
    int chk;

    ABCDK_ASSERT(size > 0 && connect_cb != NULL, "池大小不能为0，并且连接回调函数指针不能为空。");

    p = (abcdk_odbcpool_t*)abcdk_heap_alloc(sizeof(abcdk_odbcpool_t));
    if(!p)
        return NULL;

    p->mutex = abcdk_mutex_create();

    p->pool = abcdk_pool_create(sizeof(abcdk_odbc_t*),size);
    p->magic = abcdk_atomic_fetch_and_add(&magic,1);
    p->pool_size = size;
    p->exitflag = 0;
    p->pop_nbs = 0;
    p->connect_cb = connect_cb;
    p->opaque = opaque;

    return p;
}


abcdk_odbc_t *abcdk_odbcpool_pop(abcdk_odbcpool_t *ctx,time_t timeout)
{
    abcdk_odbcpool_t *p = NULL;
    abcdk_odbc_t *odbc_p = NULL;
    time_t time_end;
    time_t time_span;
    int chk;

    assert(ctx != NULL && timeout > 0);

    p = ctx;

    /*计算过期时间。*/
    time_end = _abcdk_odbcpool_clock() + timeout;

    abcdk_mutex_lock(p->mutex,1);

    while(!p->exitflag)
    {
        /*尝试从池中取一个连接，如果成功则直接跳出，否则尝试创建新的或等待回收复用。*/
        odbc_p = NULL;
        chk = abcdk_pool_pull(p->pool,&odbc_p);
        if(chk == 0)
            break;

        /*计算剩余超时时长。*/
        time_span = time_end - _abcdk_odbcpool_clock();
        if (time_span <= 0)
            break;

        /*如果未达到连接池上限，则创建新的连接。*/
        if(p->pop_nbs < p->pool_size)
        {
            odbc_p = abcdk_odbc_alloc(p->magic);
            if(!odbc_p)
                break;

            chk = p->connect_cb(odbc_p, p->opaque);
            if (chk == 0)
                break;
            else 
                abcdk_odbc_free(&odbc_p);
        }
        else
        {
            /*所有连接都已经弹出，等待连接回收复用。*/
            abcdk_mutex_wait(p->mutex,time_span);
        }
    }

    /*累加弹出数量(非常重要)，线程池销毁前要根据此值确定是否还有未回收的连接。*/
    if(odbc_p)
        p->pop_nbs += 1;

    /*唤醒其它线程，处理事件。*/
    abcdk_mutex_signal(p->mutex, 0);
    
    abcdk_mutex_unlock(p->mutex);

    return odbc_p;
}

void abcdk_odbcpool_push(abcdk_odbcpool_t *ctx, abcdk_odbc_t **odbc)
{
    abcdk_odbcpool_t *p = NULL;
    abcdk_odbc_t *odbc_p = NULL;
    int chk = -1;

    assert(ctx != NULL && odbc != NULL);
    
    ABCDK_ASSERT(*odbc != NULL,"无效的连接。");

    p = ctx;

    /*复制并清空。*/
    odbc_p = *odbc;
    *odbc = NULL;

    ABCDK_ASSERT(p->magic == abcdk_odbc_get_pool(odbc_p),"不属于当前连池。");

    abcdk_mutex_lock(p->mutex,1);

    if (p->pop_nbs > 0)
    {
        chk = abcdk_pool_push(p->pool, &odbc_p);
        if (chk == 0)
        {
            /*递减弹出数量(非常重要)，线程池销毁前要根据此值确定是否还有未回收的连接。*/
            p->pop_nbs -= 1;
            
            /*通知pop有可用连接。*/
            abcdk_mutex_signal(p->mutex,0);
        }
    }

    abcdk_mutex_unlock(p->mutex);

    ABCDK_ASSERT(chk == 0,"连接池已满，可能有不属于这个连接池的连接已经被回收。");
}

#endif // defined(__SQL_H) && defined(__SQLEXT_H)
