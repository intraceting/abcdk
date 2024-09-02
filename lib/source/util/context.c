/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/context.h"

/**简单的上下文环境。 */
struct _abcdk_context 
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_CONTEXT_MAGIC 123456789

    /** 引用计数器。*/
    volatile int refcount;

    /**同步锁类型。 */
    int sync_type;

    /**同步锁。*/
    union 
    {
        void *non_ctx;
        abcdk_mutex_t *mutex_ctx;
        abcdk_spinlock_t *spin_ctx;
        abcdk_rwlock_t *rw_ctx;
    }sync;

    /** 用户环境指针。*/
    abcdk_object_t *userdata;

    /** 用户环境销毁函数。*/
    void (*userdata_free_cb)(void *userdata);
    
    /**读句柄。*/
    int rfd;

    /**写句柄。*/
    int wfd;

};//abcdk_context_t

void abcdk_context_unref(abcdk_context_t **ctx)
{
    abcdk_context_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->magic == ABCDK_CONTEXT_MAGIC);

    if (abcdk_atomic_fetch_and_add(&ctx_p->refcount, -1) != 1)
        return;

    assert(ctx_p->refcount == 0);

    ctx_p->magic = 0xcccccccc;

    if(ctx_p->userdata_free_cb)
        ctx_p->userdata_free_cb(ctx_p->userdata->pptrs[0]);

    abcdk_object_unref(&ctx_p->userdata);

    if(ctx_p->sync_type == ABCDK_CONTEXT_SYNC_MUTEX)
        abcdk_mutex_destroy(&ctx_p->sync.mutex_ctx);
    else if(ctx_p->sync_type == ABCDK_CONTEXT_SYNC_SPINLOCK)
        abcdk_spinlock_destroy(&ctx_p->sync.spin_ctx);
    else if(ctx_p->sync_type == ABCDK_CONTEXT_SYNC_RWLOCK)
        abcdk_rwlock_destroy(&ctx_p->sync.rw_ctx);

    abcdk_closep(&ctx_p->rfd);
    abcdk_closep(&ctx_p->wfd);
    abcdk_heap_free(ctx_p);
}

abcdk_context_t *abcdk_context_refer(abcdk_context_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_context_t *abcdk_context_alloc(int sync_type, size_t userdata, void (*free_cb)(void *userdata))
{
    abcdk_context_t *ctx;

    assert(sync_type == ABCDK_CONTEXT_SYNC_NON || sync_type == ABCDK_CONTEXT_SYNC_MUTEX ||
           sync_type == ABCDK_CONTEXT_SYNC_SPINLOCK || sync_type == ABCDK_CONTEXT_SYNC_RWLOCK);

    ctx = abcdk_heap_alloc(sizeof(abcdk_context_t));
    if (!ctx)
        return NULL;

    ctx->magic = ABCDK_CONTEXT_MAGIC;
    ctx->refcount = 1;

    ctx->sync_type = sync_type;

    if(ctx->sync_type == ABCDK_CONTEXT_SYNC_MUTEX)
        ctx->sync.mutex_ctx = abcdk_mutex_create();
    else if(ctx->sync_type == ABCDK_CONTEXT_SYNC_SPINLOCK)
        ctx->sync.spin_ctx = abcdk_spinlock_create();
    else if(ctx->sync_type == ABCDK_CONTEXT_SYNC_RWLOCK)
        ctx->sync.rw_ctx = abcdk_rwlock_create();
    else 
        ctx->sync.non_ctx = NULL;

    if(ctx->sync_type && ctx->sync.non_ctx == NULL)
        goto ERR;

    ctx->userdata = abcdk_object_alloc2(userdata);
    if(!ctx->userdata)
        goto ERR;

    ctx->rfd = ctx->wfd = -1;

    return ctx;

ERR:

    abcdk_context_unref(&ctx);
    return NULL;
}

void *abcdk_context_get_userdata(abcdk_context_t *ctx)
{
    assert(ctx != NULL);

    return ctx->userdata->pptrs[0];
}

void abcdk_context_set_fd(abcdk_context_t *ctx,int fd, int flag)
{
    assert(ctx != NULL && fd >= 0);

    if (flag == ABCDK_CONTEXT_FD_RW)
        ctx->rfd = ctx->wfd = fd;
    else if (flag == ABCDK_CONTEXT_FD_RD)
        ctx->rfd = fd;
    else if (flag == ABCDK_CONTEXT_FD_WR)
        ctx->wfd = fd;
    else
        assert(flag == ABCDK_CONTEXT_FD_RW || flag == ABCDK_CONTEXT_FD_RD || flag == ABCDK_CONTEXT_FD_WR);
}

int abcdk_context_get_fd(abcdk_context_t *ctx,int flag)
{
    assert(ctx != NULL);

    if (flag == ABCDK_CONTEXT_FD_RW)
    {
        if (ctx->rfd == ctx->wfd)
            return ctx->wfd;
        else
            return -1;
    }
    else if (flag == ABCDK_CONTEXT_FD_RD)
    {
        return ctx->rfd;
    }
    else if (flag == ABCDK_CONTEXT_FD_WR)
    {
        return ctx->wfd;
    }
    else
        assert(flag == ABCDK_CONTEXT_FD_RW || flag == ABCDK_CONTEXT_FD_RD || flag == ABCDK_CONTEXT_FD_WR);

    return -1;
}

void abcdk_context_unlock(abcdk_context_t *ctx)
{
    assert(ctx != NULL);

    if(ctx->sync_type == ABCDK_CONTEXT_SYNC_MUTEX)
        abcdk_mutex_unlock(ctx->sync.mutex_ctx);
    else if(ctx->sync_type == ABCDK_CONTEXT_SYNC_SPINLOCK)
        abcdk_spinlock_unlock(ctx->sync.spin_ctx);
    else if(ctx->sync_type == ABCDK_CONTEXT_SYNC_RWLOCK)
        abcdk_rwlock_unlock(ctx->sync.rw_ctx);
    else
        ABCDK_ASSERT(0,"不支持或未创建。");
}

void abcdk_context_lock(abcdk_context_t *ctx)
{
    assert(ctx != NULL);

    if(ctx->sync_type == ABCDK_CONTEXT_SYNC_MUTEX)
        abcdk_mutex_lock(ctx->sync.mutex_ctx, 1);
    else if(ctx->sync_type == ABCDK_CONTEXT_SYNC_SPINLOCK)
        abcdk_spinlock_lock(ctx->sync.spin_ctx,1);
    else if(ctx->sync_type == ABCDK_CONTEXT_SYNC_RWLOCK)
        abcdk_rwlock_wrlock(ctx->sync.rw_ctx,1);
    else 
        ABCDK_ASSERT(0,"不支持或未创建。");
}

void abcdk_context_rdlock(abcdk_context_t *ctx)
{
    assert(ctx != NULL);

    if(ctx->sync_type == ABCDK_CONTEXT_SYNC_RWLOCK)
        abcdk_rwlock_rdlock(ctx->sync.rw_ctx,1);
    else 
        ABCDK_ASSERT(0,"不支持或未创建。");
}

void abcdk_context_wrlock(abcdk_context_t *ctx)
{
    assert(ctx != NULL);

    if(ctx->sync_type == ABCDK_CONTEXT_SYNC_RWLOCK)
        abcdk_rwlock_wrlock(ctx->sync.rw_ctx,1);
    else 
        ABCDK_ASSERT(0,"不支持或未创建。");
}

void abcdk_context_signal(abcdk_context_t *ctx,int broadcast)
{
    assert(ctx != NULL);

    if(ctx->sync_type == ABCDK_CONTEXT_SYNC_MUTEX)
        abcdk_mutex_signal(ctx->sync.mutex_ctx,broadcast);
    else 
        ABCDK_ASSERT(0,"不支持或未创建。");
}

int abcdk_context_wait(abcdk_context_t *ctx, time_t timeout)
{
    int chk = -1;

    assert(ctx != NULL);

    if(ctx->sync_type == ABCDK_CONTEXT_SYNC_MUTEX)
        chk = abcdk_mutex_wait(ctx->sync.mutex_ctx, timeout);
    else 
        ABCDK_ASSERT(0,"不支持或未创建。");

    return chk;
}