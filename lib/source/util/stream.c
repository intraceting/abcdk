/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/stream.h"

/**简单的数据流。*/
struct _abcdk_stream
{
    /** 队列。*/
    abcdk_tree_t *queue;

    /** 锁。*/
    abcdk_mutex_t *locker;

    /** 游标。*/
    size_t pos;
}; // abcdk_stream_t;

void abcdk_stream_destroy(abcdk_stream_t **ctx)
{
    abcdk_stream_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_tree_free(&ctx_p->queue);
    abcdk_mutex_destroy(&ctx_p->locker);
    abcdk_heap_free(ctx_p);
}

abcdk_stream_t *abcdk_stream_create()
{
    abcdk_stream_t *ctx;

    ctx = (abcdk_stream_t*)abcdk_heap_alloc(sizeof(abcdk_stream_t));
    if(!ctx)
        return NULL;

    ctx->queue = abcdk_tree_alloc(NULL);
    ctx->locker = abcdk_mutex_create();
    ctx->pos = 0;

    return ctx;
}

ssize_t abcdk_stream_read(abcdk_stream_t *ctx,void *buf,size_t len)
{
    abcdk_tree_t *p;
    ssize_t rlen, rall = 0;

    assert(ctx != NULL && buf != NULL && len >0);

NEXT_NODE:

    /*从队列头部开始发送。*/
    abcdk_mutex_lock(ctx->locker,1);
    p = abcdk_tree_child(ctx->queue,1);
    abcdk_mutex_unlock(ctx->locker);

    if(!p)
        return rall;

    rlen = ABCDK_MIN((size_t)len-rall,(size_t)(p->obj->sizes[0] - ctx->pos));
    memcpy(ABCDK_PTR2VPTR(buf,rall),ABCDK_PTR2VPTR(p->obj->pptrs[0], ctx->pos),rlen);

    /*滚动游标。*/
    ctx->pos += rlen;
    rall += rlen;

    /*当前节点未读完整，直接返回。*/
    if (ctx->pos < p->obj->sizes[0])
        return rall;

    /*游标归零。*/
    ctx->pos = 0;

    /*从队列中删除已经读完整的节点。*/
    abcdk_mutex_lock(ctx->locker,1);
    abcdk_tree_unlink(p);
    abcdk_tree_free(&p);
    abcdk_mutex_unlock(ctx->locker);

    /*继续读剩余节点。*/
    goto NEXT_NODE;
}

int abcdk_stream_write(abcdk_stream_t *ctx,abcdk_object_t *data)
{
    abcdk_tree_t *new_node;

    assert(ctx != NULL && data != NULL);

    new_node = abcdk_tree_alloc(data);
    if(!new_node)
        return -1;

    abcdk_mutex_lock(ctx->locker,1);
    abcdk_tree_insert2(ctx->queue,new_node,0);
    abcdk_mutex_unlock(ctx->locker);

    return 0;
}

int abcdk_stream_write_buffer(abcdk_stream_t *ctx,const void *buf,size_t len)
{
    abcdk_object_t *obj;
    int chk;

    assert(ctx != NULL && buf != NULL && len >0);

    obj = abcdk_object_copyfrom(buf,len);
    if(!obj)
        return -1;

    chk = abcdk_stream_write(ctx,obj);
    if(chk == 0)
        return 0;

    /*删除写入失败的。*/
    abcdk_object_unref(&obj);
    return -1;
}
