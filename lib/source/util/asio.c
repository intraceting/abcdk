/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/asio.h"

/**异步IO对象(内部)。*/
typedef struct _abcdk_asio_internal
{
    /**epoll句柄。*/
    int epoll_fd;

    /**节点列表。*/
    abcdk_tree_t *node_list;

    /**节点列表锁。*/
    abcdk_rwlock_t *node_locker;
    
}abcdk_asio_internal_t;

/**异步IO节点(内部)。*/
typedef struct _abcdk_asio_node_internal
{
    /*环境指针。*/
    abcdk_asio_t *ctx;

    /*用户数据。*/
    abcdk_object_t *userdata;
    void (*userdata_free_cb)(void *userdata);

    /**同步锁。*/
    abcdk_spinlock_t *sync_locker;

    /** 句柄。*/
    int fd;

    /** 状态。!0 正常，0 异常。 */
    int stable;

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

} abcdk_asio_node_internal_t;

static time_t _abcdk_asio_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC,3);
}

static int _abcdk_asio_node_add(abcdk_asio_t *ctx,abcdk_asio_node_t *node_ctx)
{
    abcdk_asio_internal_t *ctx_in_p;
    abcdk_asio_node_internal_t *node_in_p;
    abcdk_tree_t *list_node_ctx;

    ctx_in_p = (abcdk_asio_internal_t*) abcdk_context_get_userdata((abcdk_context_t *)ctx);
    node_in_p = (abcdk_asio_node_internal_t*)abcdk_context_get_userdata((abcdk_context_t *)node_ctx);

    list_node_ctx = abcdk_tree_alloc3(0);
    if(!list_node_ctx)
        return -1;

    /*引用后复制。*/
    list_node_ctx->obj->pptrs[0] = abcdk_context_refer((abcdk_context_t *)node_ctx);

    abcdk_rwlock_wrlock(ctx_in_p->node_locker,1);
    abcdk_tree_insert2(ctx_in_p->node_list,list_node_ctx,0);
    abcdk_rwlock_unlock(ctx_in_p->node_locker);

    return 0;
}


/**释放。 */
void abcdk_asio_unref(abcdk_asio_node_t **node_ctx)
{
    abcdk_context_unref((abcdk_context_t **)node_ctx);
}

/** 引用。*/
abcdk_asio_node_t *abcdk_asio_refer(abcdk_asio_node_t *node_ctx)
{
    return (abcdk_asio_node_t *)abcdk_context_refer((abcdk_context_t *)node_ctx);
}

static void _abcdk_asio_node_free_cb(void *userdata)
{
    abcdk_asio_node_internal_t *ctx_p = (abcdk_asio_node_internal_t*)userdata;

    if(ctx_p->userdata_free_cb)
        ctx_p->userdata_free_cb(ctx_p->userdata->pptrs[0]);

    abcdk_object_unref(&ctx_p->userdata);
    abcdk_spinlock_destroy(&ctx_p->sync_locker);
    abcdk_asio_destroy(&ctx_p->ctx);
}

/**申请。 */
abcdk_asio_node_t *abcdk_asio_alloc(abcdk_asio_t *ctx,size_t userdata, void (*free_cb)(void *userdata))
{
    abcdk_context_t *node_ctx;
    abcdk_asio_internal_t *ctx_in_p;
    abcdk_asio_node_internal_t *node_in_p;

    assert(ctx != NULL);

    ctx_in_p = (abcdk_asio_internal_t*) abcdk_context_get_userdata((abcdk_context_t *)ctx);

    node_ctx = abcdk_context_alloc(sizeof(abcdk_asio_node_internal_t),_abcdk_asio_node_free_cb);
    if(!node_ctx)
        return NULL;

    node_in_p = (abcdk_asio_node_internal_t*)abcdk_context_get_userdata(node_ctx);

    node_in_p->ctx = (abcdk_asio_t*)abcdk_context_refer(ctx);

    node_in_p->userdata = abcdk_object_alloc2(userdata);
    if(!node_in_p->userdata)
        goto ERR;

    node_in_p->userdata_free_cb = free_cb;

    node_in_p->sync_locker = abcdk_spinlock_create();
    if(!node_in_p->sync_locker)
        goto ERR;

    node_in_p->fd = -1;
    node_in_p->mark_first = 1;
    node_in_p->stable = 0;
    node_in_p->timeout = -1;
    node_in_p->active = _abcdk_asio_clock();

    return (abcdk_asio_node_t*)node_ctx;

ERR:

    abcdk_asio_unref(&node_ctx);
    return NULL;
}

void abcdk_asio_set_timeout(abcdk_asio_node_t *node_ctx, time_t timeout)
{
    abcdk_asio_node_internal_t *node_in_p;

    assert(node_ctx != NULL);

    node_in_p = (abcdk_asio_node_internal_t*)abcdk_context_get_userdata((abcdk_context_t *)node_ctx);
    
    abcdk_spinlock_lock(node_in_p->sync_locker,1);
    node_in_p->timeout = timeout;
    abcdk_spinlock_unlock(node_in_p->sync_locker);
}

void abcdk_asio_set_fd(abcdk_asio_node_t *node_ctx, int fd)
{
    abcdk_asio_node_internal_t *node_in_p;

    assert(node_ctx != NULL);

    node_in_p = (abcdk_asio_node_internal_t*)abcdk_context_get_userdata((abcdk_context_t *)node_ctx);
    
    abcdk_spinlock_lock(node_in_p->sync_locker,1);
    node_in_p->fd = fd;
    abcdk_spinlock_unlock(node_in_p->sync_locker); 
}

int abcdk_asio_mark(abcdk_asio_node_t *node_ctx, uint32_t want, uint32_t done)
{
    abcdk_asio_internal_t *ctx_in_p;
    abcdk_asio_node_internal_t *node_in_p;
    abcdk_epoll_event_t mark_event = {0};
    int chk;

    assert(node_ctx != NULL);
    assert((want & ~(ABCDK_EPOLL_INPUT | ABCDK_EPOLL_OUTPUT)) == 0); // 不允许注册出错事件。
    assert((done & ~(ABCDK_EPOLL_INPUT | ABCDK_EPOLL_OUTPUT)) == 0); // 完成事件中不能包含出错事件。

    node_in_p = (abcdk_asio_node_internal_t *)abcdk_context_get_userdata((abcdk_context_t *)node_ctx);
    ctx_in_p = (abcdk_asio_internal_t *)abcdk_context_get_userdata((abcdk_context_t *)node_in_p->ctx);

    abcdk_spinlock_lock(node_in_p->sync_locker,1);

    /*清除分派的事件(不会清除出错事件)。*/
    node_in_p->event_disp &= ~done;

    /*绑定关注的事件，如果事件没有被激活，这里需要继续绑定。*/
    node_in_p->event_mark |= want;

    /*如果未发生错误，进入正常流程。*/
    if (node_in_p->stable)
    {
        mark_event.events = node_in_p->event_mark;
        mark_event.data.ptr = node_ctx;

        if (abcdk_epoll_mark(ctx_in_p->epoll_fd, node_in_p->fd, &mark_event, node_in_p->mark_first) != 0)
            node_in_p->stable = 0;

        /*无论是否成功，第一次注册都已经完成。*/
        node_in_p->mark_first = 0;

        /*更节点新活动时间。*/
        node_in_p->active = _abcdk_epollex_clock();
    }

    chk = (node_in_p->stable ? 0 : -1);

    abcdk_spinlock_unlock(node_in_p->sync_locker);

    return chk;
}

void abcdk_asio_destroy(abcdk_asio_t **ctx)
{
    abcdk_context_unref((abcdk_context_t**)ctx);
}

static void _abcdk_asio_free_cb(void *userdata)
{
    abcdk_asio_internal_t *ctx_p = (abcdk_asio_internal_t*)userdata;

    abcdk_tree_free(&ctx_p->node_list);
    abcdk_rwlock_destroy(&ctx_p->node_locker);
    abcdk_closep(&ctx_p->epoll_fd);
}

/*创建。*/
abcdk_asio_t *abcdk_asio_create()
{
    abcdk_context_t *ctx;
    abcdk_asio_internal_t *ctx_in_p;

    ctx = abcdk_context_alloc(sizeof(abcdk_asio_internal_t),_abcdk_asio_free_cb);
    if(!ctx)
        return NULL;

    ctx_in_p = (abcdk_asio_internal_t*) abcdk_context_get_userdata(ctx);

    ctx_in_p->node_list = abcdk_tree_alloc(NULL);
    if(!ctx_in_p->node_list)
        goto ERR;

    ctx_in_p->node_locker = abcdk_rwlock_create();
    if(!ctx_in_p->node_locker)
        goto ERR;
    

    return (abcdk_asio_t*)ctx;

ERR:

    abcdk_asio_destroy(&ctx);
    return NULL;
}

int abcdk_asio_wait(abcdk_asio_t *ctx,abcdk_epoll_event_t events[20],time_t timeout)
{
    abcdk_asio_internal_t *ctx_in_p;
    abcdk_epoll_event_t events_tmp[20] = {0};
    int chk;

    assert(ctx != NULL && events != NULL);

    ctx_in_p = (abcdk_asio_internal_t *)abcdk_context_get_userdata((abcdk_context_t *)ctx);

    chk = abcdk_epoll_wait(ctx_in_p->epoll_fd, events_tmp,20,timeout);
    if(chk <0)
        return -1;
    else if(chk == 0)
        return 0;


    return 1;
}