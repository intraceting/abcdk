/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/stream.h"

/**简单的数据流.*/
struct _abcdk_stream
{
    /** 魔法数.*/
    uint32_t magic;
#define ABCDK_STREAM_MAGIC 123456789

    /** 引用计数器.*/
    volatile int refcount;

    /** 队列.*/
    abcdk_tree_t *queue;

    /** 锁.*/
    abcdk_mutex_t *locker;

    /** 读片段的游标.*/
    size_t read_segment_pos;

    /** 读和写总数量. */
    size_t read_total_size;
    size_t write_total_size;

}; // abcdk_stream_t;

void abcdk_stream_destroy(abcdk_stream_t **ctx)
{
   abcdk_stream_unref(ctx);
}

void abcdk_stream_unref(abcdk_stream_t **ctx)
{
    abcdk_stream_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->magic == ABCDK_STREAM_MAGIC);

    if (abcdk_atomic_fetch_and_add(&ctx_p->refcount, -1) != 1)
        return;

    assert(ctx_p->refcount == 0);

    ctx_p->magic = 0xcccccccc;

    abcdk_tree_free(&ctx_p->queue);
    abcdk_mutex_destroy(&ctx_p->locker);
    abcdk_heap_free(ctx_p);
}

abcdk_stream_t *abcdk_stream_refer(abcdk_stream_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_stream_t *abcdk_stream_create()
{
    abcdk_stream_t *ctx;

    ctx = (abcdk_stream_t*)abcdk_heap_alloc(sizeof(abcdk_stream_t));
    if(!ctx)
        return NULL;

    ctx->magic = ABCDK_STREAM_MAGIC;
    ctx->refcount = 1;

    ctx->queue = abcdk_tree_alloc(NULL);
    ctx->locker = abcdk_mutex_create();
    ctx->read_segment_pos = 0;
    ctx->read_total_size = 0;
    ctx->write_total_size = 0;

    return ctx;
}

size_t abcdk_stream_tell(abcdk_stream_t *ctx,int writer)
{
    assert(ctx != NULL);

    if(writer)
        return ctx->write_total_size;

    return ctx->read_total_size;
}

ssize_t abcdk_stream_read(abcdk_stream_t *ctx,void *buf,size_t len)
{
    abcdk_tree_t *p;
    ssize_t rlen, rall = 0;

    assert(ctx != NULL && buf != NULL && len >0);

NEXT_NODE:

    /*从队列头部开始发送.*/
    abcdk_mutex_lock(ctx->locker,1);
    p = abcdk_tree_child(ctx->queue,1);
    abcdk_mutex_unlock(ctx->locker);

    if(!p)
        return rall;

    rlen = ABCDK_MIN((size_t)len-rall,(size_t)(p->obj->sizes[0] - ctx->read_segment_pos));
    memcpy(ABCDK_PTR2VPTR(buf,rall),ABCDK_PTR2VPTR(p->obj->pptrs[0], ctx->read_segment_pos),rlen);

    /*滚动游标.*/
    ctx->read_segment_pos += rlen;
    rall += rlen;

    /*累加计数.*/
    ctx->read_total_size += rlen;

    /*当前节点未读完整, 直接返回.*/
    if (ctx->read_segment_pos < p->obj->sizes[0])
        return rall;

    /*游标归零.*/
    ctx->read_segment_pos = 0;

    /*从队列中删除已经读完整的节点.*/
    abcdk_mutex_lock(ctx->locker,1);
    abcdk_tree_unlink(p);
    abcdk_tree_free(&p);
    abcdk_mutex_unlock(ctx->locker);

    /*继续读剩余节点.*/
    goto NEXT_NODE;
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

    /*删除写入失败的.*/
    abcdk_object_unref(&obj);
    return -1;
}

int abcdk_stream_write(abcdk_stream_t *ctx,abcdk_object_t *data)
{
    abcdk_tree_t *new_node;

    assert(ctx != NULL && data != NULL);

    new_node = abcdk_tree_alloc(data);
    if(!new_node)
        return -1;

    /*在这里累加计数不用进入锁.非常重要, 因为节点加入队列后, 有可能被其它线程消费掉.*/
    ctx->write_total_size += new_node->obj->sizes[0];

    abcdk_mutex_lock(ctx->locker,1);
    abcdk_tree_insert2(ctx->queue,new_node,0);
    abcdk_mutex_unlock(ctx->locker);


    return 0;
}