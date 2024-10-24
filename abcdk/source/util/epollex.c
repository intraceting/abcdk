/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/epollex.h"

/** epoll扩展对象。*/
struct _abcdk_epollex
{
    /** epoll句柄。 >= 0 有效。*/
    int efd;

    /** 互斥量。*/
    abcdk_mutex_t *mutex;

    /** 计数器。*/
    size_t counter;

    /** 节点表。*/
    abcdk_map_t *node_map;

    /** 事件池。*/
    abcdk_pool_t *event_pool;

    /** WAIT主线程ID。*/
    volatile pthread_t wait_leader;

    /** 看门狗活动时间(毫秒)。*/
    time_t watchdog_active;

    /** 看门狗活动间隔(毫秒)。*/
    time_t watchdog_intvl;

    /** 广播事件。*/
    uint32_t broadcast_want;

    /** 清理回调函数。*/
    abcdk_epollex_cleanup_cb cleanup_cb;

    /* 清理环境指针。*/
    void *opaque;
    
};// abcdk_epollex_t;

/** epoll节点。*/
typedef struct _abcdk_epollex_node
{
    /** 句柄。>= 0 有效。*/
    int fd;

    /** 关联数据。*/
    epoll_data_t data;

    /** 状态。!0 正常，0 异常。 */
    int stable;

    /** 注册事件。*/
    uint32_t event_mark;

    /** 分派事件。*/
    uint32_t event_disp;

    /** 引用计数。*/
    int refcount;

    /** 活动时间(毫秒)。*/
    time_t active;

    /** 超时(毫秒)。*/
    volatile time_t timeout;

    /** 是否第一次注册。!0 是，0 否。*/
    int mark_first;
    
    /** 是否第一次添加。!0 是，0 否。*/
    int add_first;


} abcdk_epollex_node_t;

time_t _abcdk_epollex_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC,3);
}

void abcdk_epollex_free(abcdk_epollex_t **ctx)
{
    abcdk_epollex_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;

    abcdk_closep(&ctx_p->efd);
    abcdk_pool_destroy(&ctx_p->event_pool);
    abcdk_map_destroy(&ctx_p->node_map);
    abcdk_mutex_destroy(&ctx_p->mutex);

    /*free.*/
    abcdk_heap_free(ctx_p);

    /*Set to NULL(0).*/
    *ctx = NULL;
}

void _abcdk_epollex_destructor_cb(abcdk_object_t *p, void *opaque)
{
    abcdk_epollex_t *ctx = (abcdk_epollex_t *)opaque;
    abcdk_epollex_node_t *node = (abcdk_epollex_node_t *)p->pptrs[ABCDK_MAP_VALUE];

    ctx->cleanup_cb(&node->data,ctx->opaque);
}

abcdk_epollex_t *abcdk_epollex_alloc(abcdk_epollex_cleanup_cb cleanup_cb, void *opaque)
{
    int efd = -1;
    abcdk_epollex_t *ctx = NULL;
    
    efd = abcdk_epoll_create();
    if (efd < 0)
        goto final_error;

    ctx = abcdk_heap_alloc(sizeof(abcdk_epollex_t));
    if(!ctx)
        goto final_error;

    ctx->efd = efd;
    ctx->counter = 0;
    ctx->event_pool = abcdk_pool_create(sizeof(abcdk_epoll_event_t), 1000);
    ctx->node_map = abcdk_map_create(400);
    ctx->mutex = abcdk_mutex_create();
    ctx->watchdog_intvl = 1000;
    ctx->watchdog_active = _abcdk_epollex_clock();
    ctx->wait_leader = 0;
    ctx->node_map->destructor_cb = _abcdk_epollex_destructor_cb;
    ctx->node_map->opaque = ctx;
    ctx->cleanup_cb = cleanup_cb;
    ctx->opaque = opaque;

    return ctx;

final_error:

    abcdk_closep(&efd);
    abcdk_heap_free(ctx);

    return NULL;
}

int abcdk_epollex_detach(abcdk_epollex_t *ctx,int fd)
{
    abcdk_object_t *p = NULL;
    abcdk_epollex_node_t *node = NULL;
    int chk = 0;

    assert(ctx != NULL && fd >= 0);

    abcdk_mutex_lock(ctx->mutex,1);

    p = abcdk_map_find(ctx->node_map, &fd, sizeof(fd),0);
    if(!p)
        goto final_error;

    node = (abcdk_epollex_node_t *)p->pptrs[ABCDK_MAP_VALUE];

    if (node->refcount > 0)
        ABCDK_ERRNO_AND_GOTO1(EBUSY, final_error);

    abcdk_epoll_drop(ctx->efd,fd);
    abcdk_map_remove(ctx->node_map, &fd, sizeof(fd));

    /*计数器 -1。*/
    ctx->counter -= 1;

    /*No error.*/
    goto final;

final_error:

    chk = -1;

final:

    abcdk_mutex_unlock(ctx->mutex);

    return chk;
}

int abcdk_epollex_attach(abcdk_epollex_t *ctx,int fd,const epoll_data_t *data)
{
    abcdk_object_t *p = NULL;
    abcdk_epollex_node_t *node = NULL;
    int chk = 0;

    assert(ctx != NULL && fd >= 0 && data != NULL);

    abcdk_mutex_lock(ctx->mutex,1);

    p = abcdk_map_find(ctx->node_map, &fd, sizeof(fd), sizeof(abcdk_epollex_node_t));
    if(!p)
        goto final_error;

    node = (abcdk_epollex_node_t *)p->pptrs[ABCDK_MAP_VALUE];

    if (node->add_first != 0)
        ABCDK_ERRNO_AND_GOTO1(EEXIST,final_error);

    node->fd = fd;
    node->data = *data;
    node->timeout = 30*1000;
    node->stable = 1;
    node->active = _abcdk_epollex_clock();
    node->mark_first = 1;
    node->add_first = 1;
    node->event_mark = node->event_disp = 0;
    node->refcount = 0;

    /*计数器 +1。*/
    ctx->counter += 1;

    /*No error.*/
    goto final;

final_error:

    chk = -1;

final:

    abcdk_mutex_unlock(ctx->mutex);

    return chk;
}

int abcdk_epollex_attach2(abcdk_epollex_t *ctx, int fd)
{
    epoll_data_t data;

    assert(ctx != NULL && fd >= 0);

    data.fd = fd;

    return abcdk_epollex_attach(ctx,fd,&data);
}

size_t abcdk_epollex_count(abcdk_epollex_t *ctx)
{
    size_t count = 0;

    assert(ctx != NULL);

    abcdk_mutex_lock(ctx->mutex,1);
    count = ctx->counter;
    abcdk_mutex_unlock(ctx->mutex);

    return count;
}

void _abcdk_epollex_disp(abcdk_epollex_t *ctx, abcdk_epollex_node_t *node, uint32_t event)
{
    abcdk_epoll_event_t disp = {0};

    /*如果有错误，记录到节点上。*/
    if (event & ABCDK_EPOLL_ERROR)
        node->stable = 0;

    if (node->stable)
    {
        /*在已注册事件中，排除已被分派的事件，才是需要分派的事件。*/
        disp.events = ((event & node->event_mark) & (~node->event_disp));
    }
    else
    {
        /*当错误发生后，如果计数器为0，则分派(仅分派一次)出错事件。*/
        if (node->refcount <= 0)
            disp.events = (ABCDK_EPOLL_ERROR & (~node->event_disp));
    }

    /*根据发生的事件增加计数器。*/
    if (disp.events & ABCDK_EPOLL_ERROR)
        node->refcount += 1;
    if (disp.events & ABCDK_EPOLL_INPUT)
        node->refcount += 1;
    if (disp.events & ABCDK_EPOLL_OUTPUT)
        node->refcount += 1;

    /*在节点上附加本次分派的事件。*/
    node->event_disp |= disp.events;

    /*清除即将通知的事件，注册事件只通知一次。*/
    node->event_mark &= ~disp.events;

    /*有事件时再推送到活动队列。*/
    if (disp.events)
    {
        disp.data = node->data;
        abcdk_pool_push(ctx->event_pool,&disp);
    }   
}

void _abcdk_epollex_mark(abcdk_epollex_t *ctx, abcdk_epollex_node_t *node, uint32_t want, uint32_t done)
{
    abcdk_epoll_event_t tmp = {0};

    /*清除分派的事件(不会清除出错事件)。*/
    node->event_disp &= ~done;

    /*绑定关注的事件，如果事件没有被激活，这里需要继续绑定。*/
    node->event_mark |= want;

    /*如果未发生错误，进入正常流程。*/
    if (node->stable)
    {
        tmp.events = node->event_mark;
        tmp.data.fd = node->fd;

        if (abcdk_epoll_mark(ctx->efd,node->fd,&tmp,node->mark_first) != 0)
            node->stable = 0;
        
        /*无论是否成功，第一次注册都已经完成。*/
        node->mark_first = 0;

        /*更节点新活动时间。*/
        node->active = _abcdk_epollex_clock();
    }
    
    /* 如果发生错误，分派出错事件。*/
    if (!node->stable)
        _abcdk_epollex_disp(ctx, node, ABCDK_EPOLL_ERROR);
}

int _abcdk_epollex_mark_scan_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_epollex_t *ctx = (abcdk_epollex_t *)opaque;
    abcdk_epollex_node_t *node = (abcdk_epollex_node_t *)alloc->pptrs[ABCDK_MAP_VALUE];

    _abcdk_epollex_mark(ctx,node,ctx->broadcast_want,0);

    return 1;
}

int abcdk_epollex_timeout(abcdk_epollex_t *ctx, int fd,time_t timeout)
{
    abcdk_object_t *p = NULL;
    abcdk_epollex_node_t *node = NULL;
    int chk = 0;

    assert(ctx != NULL && fd >= 0);

    abcdk_mutex_lock(ctx->mutex,1);

    p = abcdk_map_find(ctx->node_map, &fd, sizeof(fd),0);
    if(!p)
        goto final_error;

    node = (abcdk_epollex_node_t *)p->pptrs[ABCDK_MAP_VALUE];
    
    node->timeout = timeout;
    //assert(node->timeout != 1);

    /* 如果发生错误，分派出错事件。*/
    if (!node->stable)
        _abcdk_epollex_disp(ctx, node, ABCDK_EPOLL_ERROR);

    /*No error.*/
    goto final;

final_error:

    chk = -1;

final:

    abcdk_mutex_unlock(ctx->mutex);

    return chk; 
}

int abcdk_epollex_mark(abcdk_epollex_t *ctx, int fd, uint32_t want, uint32_t done)
{
    abcdk_object_t *p = NULL;
    abcdk_epollex_node_t *node = NULL;

    int chk = 0;

    assert(ctx != NULL);
    assert((want & ~(ABCDK_EPOLL_INPUT | ABCDK_EPOLL_OUTPUT)) == 0);//不允许注册出错事件。
    assert((done & ~(ABCDK_EPOLL_INPUT | ABCDK_EPOLL_OUTPUT)) == 0);//完成事件中不能包含出错事件。

    abcdk_mutex_lock(ctx->mutex,1);

    if (fd >= 0)
    {
        p = abcdk_map_find(ctx->node_map, &fd, sizeof(fd), 0);
        if (!p)
            goto final_error;

        node = (abcdk_epollex_node_t *)p->pptrs[ABCDK_MAP_VALUE];

        _abcdk_epollex_mark(ctx, node, want, done);
    }
    else
    {
        /*Set.*/
        ctx->broadcast_want = want;

        /*遍历。*/
        ctx->node_map->dump_cb = _abcdk_epollex_mark_scan_cb;
        ctx->node_map->opaque = ctx;
        abcdk_map_scan(ctx->node_map);

        /*Clear.*/
        ctx->broadcast_want = 0;
    }

    /*No error.*/
    goto final;

final_error:

    chk = -1;

final:

    abcdk_mutex_unlock(ctx->mutex);

    return chk;   
}

int _abcdk_epollex_watchdog_scan_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_epollex_t *ctx = (abcdk_epollex_t *)opaque;
    abcdk_epollex_node_t *node = (abcdk_epollex_node_t *)alloc->pptrs[ABCDK_MAP_VALUE];

    /*负值或零，不启用超时检查。*/
    if (node->timeout <= 0)
        goto final;

    /*当事件队列排队过长时，中断看门狗检查，优先处理队列中的事件。*/
    if (abcdk_pool_count(ctx->event_pool) >= 800)
        return -1;

    /*如果超时，派发ERROR事件。*/
    if ((ctx->watchdog_active - node->active) >= node->timeout)
        _abcdk_epollex_disp(ctx, node, ABCDK_EPOLL_ERROR);

final:

    return 1;
}

void _abcdk_epollex_watchdog(abcdk_epollex_t *ctx)
{
    time_t current = _abcdk_epollex_clock();

    /*看门狗活动间隔时间不能太密集。*/
    if ((current - ctx->watchdog_active) < ctx->watchdog_intvl)
        return;

    /*更新看门狗活动时间。*/
    ctx->watchdog_active = current;

    /*遍历。*/
    ctx->node_map->dump_cb = _abcdk_epollex_watchdog_scan_cb;
    ctx->node_map->opaque = ctx;
    abcdk_map_scan(ctx->node_map);
}

void _abcdk_epollex_wait_disp(abcdk_epollex_t *ctx,abcdk_epoll_event_t *events,int count)
{
    abcdk_epoll_event_t *e;
    abcdk_object_t *p;
    abcdk_epollex_node_t *node;

    for (int i = 0; i < count; i++)
    {
        e = &events[i];
        p = abcdk_map_find(ctx->node_map, &e->data.fd,sizeof(e->data.fd), 0);

        /*有那么一瞬间，当前返回的事件并不在锁(可能被分离)的保护范围内，因此这要做些特殊处理。*/
        if (p == NULL)
            continue;
            
        node = (abcdk_epollex_node_t *)p->pptrs[ABCDK_MAP_VALUE];

        /*派发事件。*/
        _abcdk_epollex_disp(ctx,node,e->events);

        /*更节点新活动时间*/
        node->active = _abcdk_epollex_clock();
    }
}

int abcdk_epollex_wait(abcdk_epollex_t *ctx, abcdk_epoll_event_t *event, time_t timeout)
{
    abcdk_epoll_event_t w[20];
    time_t time_end;
    time_t time_span;
    int count;
    int chk = 0, wait_chk = 0;

    assert(ctx != NULL && event != NULL && timeout > 0);

    /*计算过期时间。*/
    time_end = _abcdk_epollex_clock() + timeout;

    /*接口绑定到线程。*/
    if (abcdk_thread_leader_test(&ctx->wait_leader) != 0)
    {
        if (abcdk_thread_leader_vote(&ctx->wait_leader) != 0)
            ABCDK_ASSERT(0,"仅允许固定线程调用此接口。");
    }

    abcdk_mutex_lock(ctx->mutex, 1);

try_again:

    /*优先从事件队列中拉取，有数据直接跳转结束，无数据进入等待。*/
    chk = abcdk_pool_pull(ctx->event_pool, event);
    if (chk == 0)
        goto final;

    /*计算剩余超时时长。*/
    time_span = time_end - _abcdk_epollex_clock();
    if (time_span <= 0)
        goto final_error;

    /*通过看门狗检测长期不活动的节点。*/
    _abcdk_epollex_watchdog(ctx);

    /*如果有过期节点，则不启用IO等待时。*/
    time_span = (abcdk_pool_count(ctx->event_pool) > 0) ? (0) : ABCDK_MIN(time_span, ctx->watchdog_intvl);

    /*解锁，使其它接口被访问。*/
    abcdk_mutex_unlock(ctx->mutex);

    /*IO等待。*/
    count = abcdk_epoll_wait(ctx->efd, w, ABCDK_ARRAY_SIZE(w), ABCDK_MIN(time_span, ctx->watchdog_intvl));

    /*加锁，禁止其它接口被访问。*/
    abcdk_mutex_lock(ctx->mutex, 1);

    /*处理活动事件。*/
    _abcdk_epollex_wait_disp(ctx, w, count);

    /*No error, no event, try again.*/
    goto try_again;

final_error:

    chk = -1;

final:

    abcdk_mutex_unlock(ctx->mutex);

    return chk;
}

int abcdk_epollex_unref(abcdk_epollex_t *ctx,int fd, uint32_t events)
{
    abcdk_object_t *p = NULL;
    abcdk_epollex_node_t *node = NULL;
    abcdk_epoll_event_t tmp = {0};
    int chk = 0;

    assert(ctx != NULL && fd >= 0);

    assert((events & ~(ABCDK_EPOLL_INPUT | ABCDK_EPOLL_OUTPUT | ABCDK_EPOLL_ERROR)) == 0);

    if (abcdk_thread_leader_test(&ctx->wait_leader) != 0)
        ABCDK_ASSERT(0,"仅允许固定线程调用此接口。");

    abcdk_mutex_lock(ctx->mutex,1);

    p = abcdk_map_find(ctx->node_map, &fd, sizeof(fd),0);
    if(!p)
        goto final_error;

    node = (abcdk_epollex_node_t *)p->pptrs[ABCDK_MAP_VALUE];

    /*无论成功或失败，记数器都要相应的减少，不然无法释放。*/
    if (events & ABCDK_EPOLL_ERROR)
        node->refcount -= 1;
    if (events & ABCDK_EPOLL_INPUT)
        node->refcount -= 1;
    if (events & ABCDK_EPOLL_OUTPUT)
        node->refcount -= 1;

    assert(node->refcount >= 0);

    /* 如果发生错误，分派出错事件。*/
    if (!node->stable)
        _abcdk_epollex_disp(ctx, node, ABCDK_EPOLL_ERROR);

    /*No error.*/
    goto final;

final_error:

    chk = -1;

final:

    abcdk_mutex_unlock(ctx->mutex);

    return chk; 
}