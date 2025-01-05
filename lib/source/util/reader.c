/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include "abcdk/util/reader.h"

/** 读者环境。*/
struct _abcdk_reader
{
    /** 状态。1： 运行，2：停止。*/
    volatile int status;

    /** 工作线程。*/
    abcdk_thread_t tid;

    /** 文件结束标志。0：未结束，1：已结束。*/
    volatile int eof;

    /** 文件句柄。*/
    int fd;

    /** 块大小。*/
    size_t blksize;

    /** 忙队列。*/
    abcdk_tree_t *qbusy;

    /** 闲队列。*/
    abcdk_tree_t *qidle;

    /** 队列锁。*/
    abcdk_mutex_t *qlock;

};// abcdk_reader_t;

void abcdk_reader_destroy(abcdk_reader_t **reader)
{
    abcdk_reader_t *reader_p = NULL;

    if (reader == NULL || *reader == NULL)
        return;

    reader_p = *reader;

    abcdk_reader_stop(reader_p);

    abcdk_tree_free(&reader_p->qbusy);
    abcdk_tree_free(&reader_p->qidle);
    abcdk_mutex_destroy(&reader_p->qlock);

    abcdk_heap_free(reader_p);

    /*Set NULL(0).*/
    *reader = NULL;
}

abcdk_reader_t *abcdk_reader_create(size_t blksize)
{
    abcdk_reader_t *reader = NULL;
    abcdk_tree_t *buf;

    assert(blksize > 0);

    reader = abcdk_heap_alloc(sizeof(abcdk_reader_t));
    if (!reader)
        goto final_error;

    reader->status = 2;
    reader->tid.handle = 0;
    reader->eof = 1;
    reader->fd = -1;
    reader->blksize = blksize;

    reader->qbusy = abcdk_tree_alloc3(1);
    if (!reader->qbusy)
        goto final_error;

    reader->qidle = abcdk_tree_alloc3(1);
    if (!reader->qbusy)
        goto final_error;

    reader->qlock = abcdk_mutex_create();

    for (int i = 0; i < 100; i++)
    {
        /*缓存指针，数据长度，已读长度。*/
        size_t sizes[] = {blksize, sizeof(size_t), sizeof(size_t)};
        buf = abcdk_tree_alloc2(sizes, 3, 0);
        if (!buf)
            goto final_error;

        abcdk_tree_insert2(reader->qidle, buf, 0);
    }

    return reader;

final_error:

    abcdk_tree_free(&reader->qbusy);
    abcdk_tree_free(&reader->qidle);
    abcdk_mutex_destroy(&reader->qlock);

    abcdk_heap_free(reader);

    return NULL;
}

void abcdk_reader_stop(abcdk_reader_t *reader)
{
    assert(reader != NULL);

    if (!abcdk_atomic_compare_and_swap(&reader->status, 1, 2))
        return;

    abcdk_mutex_signal(reader->qlock, 1);
    abcdk_thread_join(&reader->tid);
}

static void *_abcdk_reader_readfile(void *opaque);

int abcdk_reader_start(abcdk_reader_t *reader, int fd)
{
    abcdk_tree_t *buf;
    int chk;

    assert(reader != NULL && fd >= 0);

    if (!abcdk_atomic_compare_and_swap(&reader->status, 2, 1))
        return -2;

    reader->tid.opaque = reader;
    reader->tid.routine = _abcdk_reader_readfile;
    reader->eof = 0;
    reader->fd = fd;

    /*忙队列节点转移到闲队列中。*/
    while (1)
    {
        buf = abcdk_tree_child(reader->qbusy, 1);
        if(!buf)
            break;
    
        abcdk_tree_unlink(buf);
        abcdk_tree_insert2(reader->qidle, buf, 0);
    }

    chk = abcdk_thread_create(&reader->tid, 1);
    if (chk == 0)
        return 0;

    /*走到这里表示线程未创建成功。*/
    abcdk_atomic_store(&reader->status, 2);
    reader->tid.opaque = NULL;
    reader->tid.routine = NULL;
    reader->eof = 1;
    reader->fd = -1;

    return -1;
}

void *_abcdk_reader_readfile(void *opaque)
{
    abcdk_reader_t *reader = (abcdk_reader_t *)opaque;
    abcdk_tree_t *buf;
    ssize_t rlen;

    while (abcdk_atomic_load(&reader->status) == 1)
    {
        /*从闲队列取一个节点。*/
        abcdk_mutex_lock(reader->qlock, 1);
        buf = abcdk_tree_child(reader->qidle, 1);
        if (buf)
            abcdk_tree_unlink(buf);
        else
            abcdk_mutex_wait(reader->qlock, -1);
        abcdk_mutex_unlock(reader->qlock);

        /*如果闲队列为空，则重新等待。*/
        if (!buf)
            continue;

        /*清除旧的信息。*/
        ABCDK_PTR2SIZE(buf->obj->pptrs[1], 0) = ABCDK_PTR2SIZE(buf->obj->pptrs[2], 0) = 0;
        /*读取新的数据。*/
        rlen = abcdk_read(reader->fd, buf->obj->pptrs[0], reader->blksize);
        if (rlen > 0)
            ABCDK_PTR2SIZE(buf->obj->pptrs[1], 0) = rlen;

        /*添加到忙队列末尾。*/
        abcdk_mutex_lock(reader->qlock, 1);
        abcdk_tree_insert2(reader->qbusy, buf, 0);
        abcdk_mutex_signal(reader->qlock, 0);
        abcdk_mutex_unlock(reader->qlock);

        /*未填满块大小，表示已经到文件末尾，退出。*/
        if (rlen < reader->blksize)
            break;
    }

    /*无论什么原因，走到这里表示文件已经结束。*/
    abcdk_atomic_store(&reader->eof,1);

    return NULL;
}

ssize_t abcdk_reader_read(abcdk_reader_t *reader, void *buf, size_t size)
{
    abcdk_tree_t *buf2;
    ssize_t rall = 0, rper = 0;

    assert(reader != NULL && buf != NULL && size > 0);

    while (abcdk_atomic_load(&reader->status) == 1)
    {
        /*从忙队列取一个节点。*/
        abcdk_mutex_lock(reader->qlock, 1);
        while (abcdk_atomic_load(&reader->status) == 1)
        {
            buf2 = abcdk_tree_child(reader->qbusy, 1);
            if (buf2)
            {
                abcdk_tree_unlink(buf2);
                break;
            }
            else if (abcdk_atomic_compare(&reader->eof,0))
                abcdk_mutex_wait(reader->qlock, -1);
            else 
                break;
        };
        abcdk_mutex_unlock(reader->qlock);

        /*没有数据，表示文件已经结束。*/
        if (!buf2)
            break;

        /*计算最多读多少。*/
        rper = ABCDK_MIN(ABCDK_PTR2SIZE(buf2->obj->pptrs[1], 0) - ABCDK_PTR2SIZE(buf2->obj->pptrs[2], 0), size - rall);
        memcpy(ABCDK_PTR2VPTR(buf, rall), buf2->obj->pptrs[0] + ABCDK_PTR2SIZE(buf2->obj->pptrs[2], 0), rper);

        /*累加已读量。*/
        ABCDK_PTR2SIZE(buf2->obj->pptrs[2], 0) += rper;
        rall += rper;

        /*判断缓存区中数据是否已经读完。*/
        if (ABCDK_PTR2SIZE(buf2->obj->pptrs[1], 0) > ABCDK_PTR2SIZE(buf2->obj->pptrs[2], 0))
        {
            /*未读完，将缓存放入忙队列(头)。*/
            abcdk_mutex_lock(reader->qlock, 1);
            abcdk_tree_insert2(reader->qbusy, buf2, 1);
            abcdk_mutex_unlock(reader->qlock);
        }
        else
        {
            /*已读完，将缓存放入闲队列(尾)。*/
            abcdk_mutex_lock(reader->qlock, 1);
            abcdk_tree_insert2(reader->qidle, buf2, 0);
            abcdk_mutex_signal(reader->qlock, 0);
            abcdk_mutex_unlock(reader->qlock);
        }

        /*数据读够了，返回退出。*/
        if (rall >= size)
            break;
    }

    return rall;
}
