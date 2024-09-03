/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/asio.h"

/**异步IO对象。*/
typedef struct _abcdk_asio
{
    /**epoll句柄。*/
    int epoll_fd;

    /**节点列表。*/
    abcdk_object_t *node_list;

    /**节点同步锁。*/
    abcdk_mutex_t *node_sync;

    /**事件池。*/
    abcdk_pool_t *event_pool;

    /**句柄索引池。*/
    abcdk_pool_t *index_pool;

    /** 看门狗活动时间(毫秒)。*/
    time_t watchdog_active;

    /** 看门狗活动间隔(毫秒)。*/
    time_t watchdog_intvl;
    
}abcdk_asio_t;

/**异步IO节点。*/
typedef struct _abcdk_asio_node
{
    /**索引。*/
    int idx;

    /**伪句柄。*/
    int64_t pfd;

    /** 句柄。*/
    int fd;

    /**用户关联数据。 */
    epoll_data_t userdata;

    /** 状态。!0 正常，0 异常。 */
    int stable;

    /** 引用计数。*/
    int refcount;

    /** 注册事件。*/
    uint32_t event_mark;

    /** 分派事件。*/
    uint32_t event_disp;

    /** 活动时间(毫秒)。*/
    time_t active;

    /** 超时(毫秒)。*/
    time_t timeout;

    /** 是否第一次注册。!0 是，0 否。*/
    int mark_first;

} abcdk_asio_node_t;

static time_t _abcdk_asio_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC,3);
}

static int64_t _abcdk_asio_idx2pfd(int idx)
{
    return  (0x7FFFFFFF00000000LL | (int64_t)idx);
}

static int _abcdk_asio_pfd2idx(int64_t pfd)
{
    return  (int)(0x00000000FFFFFFFFLL & pfd);
}

static int _abcdk_asio_pull_idle_idx(abcdk_asio_t *ctx)
{
    int idx;
    int chk;

    chk = abcdk_pool_pull(ctx->index_pool, &idx);
    if (chk != 0)
        return -1;

    return idx;
}

static int _abcdk_asio_push_idle_idx(abcdk_asio_t *ctx,int idx)
{
    int chk;

    chk = abcdk_pool_push(ctx->index_pool, &idx);
    if (chk != 0)
        return -1;

    return 0;
}

static abcdk_asio_node_t *_abcdk_asio_node_alloc(abcdk_asio_t *ctx)
{
    abcdk_asio_node_t *node_ctx;
    int64_t pfd;
    int idx;

    assert(ctx != NULL);

    idx = _abcdk_asio_pull_idle_idx(ctx);
    if(idx < 0)
        return NULL;

    node_ctx = (abcdk_asio_node_t *)ctx->node_list->pptrs[idx];

    assert(node_ctx->pfd == 0);

    pfd = _abcdk_asio_idx2pfd(idx);

    node_ctx->idx = idx;
    node_ctx->pfd = pfd;

    return node_ctx;
}

static abcdk_asio_node_t *_abcdk_asio_idx2node(abcdk_asio_t *ctx,int idx)
{
    abcdk_asio_node_t *node_ctx;

    if(idx >= ctx->node_list->numbers)
        return NULL;

    node_ctx = (abcdk_asio_node_t *)ctx->node_list->pptrs[idx];

    if(node_ctx->idx != idx)
        return NULL;  

    return node_ctx;

}

static abcdk_asio_node_t *_abcdk_asio_pfd2node(abcdk_asio_t *ctx,int64_t pfd)
{
    abcdk_asio_node_t *node_ctx;
    int idx;

    idx = _abcdk_asio_pfd2idx(pfd);
    if(idx < 0)
        return NULL;

    node_ctx = _abcdk_asio_idx2node(ctx,idx);
    if(!node_ctx)
        return NULL;

    if(node_ctx->pfd != pfd)
        return NULL;   

    return node_ctx;
}

static void _abcdk_asio_disp(abcdk_asio_t *ctx, abcdk_asio_node_t *node_ctx, uint32_t event)
{
    abcdk_epoll_event_t disp = {0};

    /*如果有错误，记录到节点上。*/
    if (event & ABCDK_EPOLL_ERROR)
        node_ctx->stable = 0;

    if (node_ctx->stable)
    {
        /*在已注册事件中，排除已被分派的事件，才是需要分派的事件。*/
        disp.events = ((event & node_ctx->event_mark) & (~node_ctx->event_disp));
    }
    else
    {
        /*当错误发生后，如果计数器为0，则分派(仅分派一次)出错事件。*/
        if (node_ctx->refcount <= 0)
            disp.events = (ABCDK_EPOLL_ERROR & (~node_ctx->event_disp));
    }

    /*根据发生的事件增加计数器。*/
    if (disp.events & ABCDK_EPOLL_ERROR)
        node_ctx->refcount += 1;
    if (disp.events & ABCDK_EPOLL_INPUT)
        node_ctx->refcount += 1;
    if (disp.events & ABCDK_EPOLL_OUTPUT)
        node_ctx->refcount += 1;

    /*在节点上附加本次分派的事件。*/
    node_ctx->event_disp |= disp.events;

    /*清除即将通知的事件，注册事件只通知一次。*/
    node_ctx->event_mark &= ~disp.events;

    /*有事件时再推送到活动队列。*/
    if (disp.events)
    {
        disp.data = node_ctx->userdata;
        abcdk_pool_push(ctx->event_pool,&disp);
    }   
}

void _abcdk_asio_mark(abcdk_asio_t *ctx, abcdk_asio_node_t *node_ctx, uint32_t want, uint32_t done)
{
    abcdk_epoll_event_t tmp = {0};

    /*清除分派的事件(不会清除出错事件)。*/
    node_ctx->event_disp &= ~done;

    /*绑定关注的事件，如果事件没有被激活，这里需要继续绑定。*/
    node_ctx->event_mark |= want;

    /*如果未发生错误，进入正常流程。*/
    if (node_ctx->stable)
    {
        tmp.events = node_ctx->event_mark;
        tmp.data.u64 = node_ctx->pfd;

        if (abcdk_epoll_mark(ctx->epoll_fd,node_ctx->fd,&tmp,node_ctx->mark_first) != 0)
            node_ctx->stable = 0;
        
        /*无论是否成功，第一次注册都已经完成。*/
        node_ctx->mark_first = 0;

        /*更节点新活动时间。*/
        node_ctx->active = _abcdk_asio_clock();
    }
    
    /* 如果发生错误，分派出错事件。*/
    if (!node_ctx->stable)
        _abcdk_asio_disp(ctx, node_ctx, ABCDK_EPOLL_ERROR);
}

static int _abcdk_asio_watchdog(abcdk_asio_t *ctx)
{
    abcdk_asio_node_t *node_ctx;
    time_t current = _abcdk_asio_clock();

    /*看门狗活动间隔时间不能太密集。*/
    if ((current - ctx->watchdog_active) < ctx->watchdog_intvl)
        goto END;

    /*更新看门狗活动时间。*/
    ctx->watchdog_active = current;

    for (int i = 0; i < ctx->node_list->numbers; i++)
    {
        node_ctx = _abcdk_asio_idx2node(ctx, i);
        if (!node_ctx)
            continue;

        /*负值或零，不启用超时检查。*/
        if (node_ctx->timeout <= 0)
            continue;

        /*当事件队列排队过长时，中断看门狗检查，优先处理队列中的事件。*/
        if (abcdk_pool_count(ctx->event_pool) >= 800)
            break;

        /*如果超时，派发ERROR事件。*/
        if ((ctx->watchdog_active - node_ctx->active) >= node_ctx->timeout)
            _abcdk_asio_disp(ctx, node_ctx, ABCDK_EPOLL_ERROR);
    }

END:

    return abcdk_pool_count(ctx->event_pool);
}

static void _abcdk_asio_lock(abcdk_asio_t *ctx)
{
    abcdk_mutex_lock(ctx->node_sync,1);
}

static int _abcdk_asio_unlock(abcdk_asio_t *ctx,int exitcode)
{
    abcdk_mutex_unlock(ctx->node_sync);

    return exitcode;
}

void abcdk_asio_destroy(abcdk_asio_t **ctx)
{
    abcdk_asio_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->node_list);
    abcdk_mutex_destroy(&ctx_p->node_sync);
    abcdk_pool_destroy(&ctx_p->event_pool);
    abcdk_pool_destroy(&ctx_p->index_pool);
    abcdk_closep(&ctx_p->epoll_fd);
    abcdk_heap_free(ctx_p);
}

abcdk_asio_t *abcdk_asio_create(int max)
{
    abcdk_asio_t *ctx;

    assert(max > 0);

    ctx = (abcdk_asio_t *)abcdk_heap_alloc(sizeof(abcdk_asio_t));
    if(!ctx)
        return NULL;

    ctx->epoll_fd = abcdk_epoll_create();
    if(ctx->epoll_fd <0)
        goto ERR;

    ctx->node_list = abcdk_object_alloc3(sizeof(abcdk_asio_node_t),max);
    if(!ctx->node_list)
        goto ERR;

    ctx->node_sync = abcdk_mutex_create();
    if(!ctx->node_sync)
        goto ERR;

    ctx->event_pool = abcdk_pool_create(sizeof(abcdk_epoll_event_t), 1000);
    if(!ctx->event_pool)
        goto ERR;

    ctx->index_pool = abcdk_pool_create(sizeof(int), max);
    if(!ctx->index_pool)
        goto ERR;

    /*初始化看门狗。*/
    ctx->watchdog_active = _abcdk_asio_clock();
    ctx->watchdog_intvl = 1000;

    /*初始化索引。*/
    for (int i = 0; i < max; i++)
        abcdk_pool_push(ctx->index_pool, &i);

    return ctx;

ERR:

    abcdk_asio_destroy(&ctx);
    return NULL;
}

int abcdk_asio_del(abcdk_asio_t *ctx,int64_t pfd)
{
    abcdk_asio_node_t *node_ctx;

    assert(ctx != NULL && pfd > 0);

    _abcdk_asio_lock(ctx);

    node_ctx = _abcdk_asio_pfd2node(ctx,pfd);
    if(!node_ctx)
        return _abcdk_asio_unlock(ctx,-22);

    if (node_ctx->refcount > 0)
        return _abcdk_asio_unlock(ctx,-16);

    node_ctx->idx = -1;
    node_ctx->pfd = 0;
    node_ctx->fd = -1;

    abcdk_epoll_drop(ctx->epoll_fd,node_ctx->fd);

    _abcdk_asio_push_idle_idx(ctx,node_ctx->idx);

    return _abcdk_asio_unlock(ctx,0);
}

int64_t abcdk_asio_add(abcdk_asio_t *ctx, int fd, epoll_data_t *userdata)
{
    abcdk_asio_node_t *node_ctx;
    int64_t pfd;

    assert(ctx != NULL && fd >= 0 && userdata != NULL);

    _abcdk_asio_lock(ctx);

    node_ctx = _abcdk_asio_node_alloc(ctx);
    if(!node_ctx)
        return _abcdk_asio_unlock(ctx,-16);

    node_ctx->fd = fd;
    node_ctx->userdata = *userdata;
    node_ctx->stable = 1;
    node_ctx->refcount = 0;
    node_ctx->event_disp = 0;
    node_ctx->event_mark = 0;
    node_ctx->active = _abcdk_asio_clock();
    node_ctx->timeout = -1;
    node_ctx->mark_first = 1;

    /*copy.*/
    pfd = node_ctx->pfd;

    _abcdk_asio_unlock(ctx,0);

    return pfd;
}

int abcdk_asio_set_timeout(abcdk_asio_t *ctx,int64_t pfd, time_t timeout)
{
    abcdk_asio_node_t *node_ctx;

    assert(ctx != NULL && pfd > 0);

    _abcdk_asio_lock(ctx);

    node_ctx = _abcdk_asio_pfd2node(ctx,pfd);
    if(!node_ctx)
        return _abcdk_asio_unlock(ctx,-22);

    node_ctx->timeout = timeout;

    /* 如果发生错误，分派出错事件。*/
    if (!node_ctx->stable)
        _abcdk_asio_disp(ctx, node_ctx, ABCDK_EPOLL_ERROR);

    return _abcdk_asio_unlock(ctx,0);
}

int abcdk_asio_mark(abcdk_asio_t *ctx,int64_t pfd,uint32_t want,uint32_t done)
{
    abcdk_asio_node_t *node_ctx;

    assert(ctx != NULL && pfd > 0);
    assert((want & ~(ABCDK_EPOLL_INPUT | ABCDK_EPOLL_OUTPUT)) == 0);//不允许注册出错事件。
    assert((done & ~(ABCDK_EPOLL_INPUT | ABCDK_EPOLL_OUTPUT)) == 0);//完成事件中不能包含出错事件。

    _abcdk_asio_lock(ctx);

    node_ctx = _abcdk_asio_pfd2node(ctx,pfd);
    if(!node_ctx)
        return _abcdk_asio_unlock(ctx,-22);

    _abcdk_asio_mark(ctx, node_ctx, want, done);

    return _abcdk_asio_unlock(ctx,0);
}

int abcdk_asio_unref(abcdk_asio_t *ctx,int64_t pfd, uint32_t events)
{
    abcdk_asio_node_t *node_ctx;

    assert(ctx != NULL && pfd > 0);

    _abcdk_asio_lock(ctx);

    node_ctx = _abcdk_asio_pfd2node(ctx,pfd);
    if(!node_ctx)
        return _abcdk_asio_unlock(ctx,-22);

    /*无论成功或失败，记数器都要相应的减少，不然无法释放。*/
    if (events & ABCDK_EPOLL_ERROR)
        node_ctx->refcount -= 1;
    if (events & ABCDK_EPOLL_INPUT)
        node_ctx->refcount -= 1;
    if (events & ABCDK_EPOLL_OUTPUT)
        node_ctx->refcount -= 1;

    assert(node_ctx->refcount >= 0);

    /* 如果发生错误，分派出错事件。*/
    if (!node_ctx->stable)
        _abcdk_asio_disp(ctx, node_ctx, ABCDK_EPOLL_ERROR);

    return _abcdk_asio_unlock(ctx,0);
}

int abcdk_asio_wait(abcdk_asio_t *ctx,abcdk_epoll_event_t *event)
{
    abcdk_asio_node_t *node_ctx;
    abcdk_epoll_event_t es_tmp[20] = {0};
    time_t iowait_ms;
    int chk;

    assert(ctx != NULL && event != NULL);

LOOP_NEXT:

    /*加锁，禁止其它接口被访问。*/
    _abcdk_asio_lock(ctx);

    /*优先从事件队列中拉取，有数据直接跳转结束，无数据进入等待。*/
    chk = abcdk_pool_pull(ctx->event_pool, event);
    if (chk == 0)
        return _abcdk_asio_unlock(ctx,1);

    /*通过看门狗检测长期不活动的节点。*/
    chk = _abcdk_asio_watchdog(ctx);

    /*根据看门狗的结果，决定IO事件等待时长。*/
    iowait_ms = (chk > 0 ? 0 : 1000);

    /*解锁，使其它接口被访问。*/
    _abcdk_asio_unlock(ctx,0);

    /*IO事件等待。*/
    chk = abcdk_epoll_wait(ctx->epoll_fd, es_tmp,20,iowait_ms);
    if(chk < 0)
        return -1;
    else if(chk == 0)
        goto LOOP_NEXT;

    /*加锁，禁止其它接口被访问。*/
    _abcdk_asio_lock(ctx);

    /*处理活动事件。*/
    for (int i = 0; i < chk; i++)
    {
        /*在多线程环境中，节点有可能被其它线程删除。*/
        node_ctx = _abcdk_asio_pfd2node(ctx, es_tmp[i].data.u64);
        if(!node_ctx)
            continue;

        _abcdk_asio_disp(ctx, node_ctx, es_tmp[i].events);
    }

    /*解锁，使其它接口被访问。*/
    _abcdk_asio_unlock(ctx,0);

    goto LOOP_NEXT;
}