/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/receiver.h"

/** 接收器对象。*/
struct _abcdk_receiver
{
    /** 引用计数器。*/
    volatile int refcount;
    
    /** 临时文件名。*/
    char tmp_file[PATH_MAX];
    
    /** 外部缓存对象。*/
    abcdk_object_t *tmp_obj;

    /** 内存块指针。*/
    void *buf;

    /* 读写偏移量。*/
    size_t offset;

    /** 容量。*/
    size_t capacity;

    /** 长度。*/
    size_t size;

    /** 消息协议。*/
    abcdk_receiver_protocol_t protocol;
    
};// abcdk_receiver_t;


void abcdk_receiver_unref(abcdk_receiver_t **ctx)
{
    abcdk_receiver_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (abcdk_atomic_fetch_and_add(&ctx_p->refcount, -1) != 1)
        return;

    assert(ctx_p->refcount == 0);

    /*创建时，如果绑定外部内部对象，则内部对象没有创建内存。*/
    if(ctx_p->tmp_obj)
        abcdk_object_unref(&ctx_p->tmp_obj);
    else
        abcdk_heap_free2(&ctx_p->buf);

    if(access(ctx_p->tmp_file,F_OK)==0)
        remove(ctx_p->tmp_file);

    abcdk_heap_free(ctx_p);
}

abcdk_receiver_t *abcdk_receiver_refer(abcdk_receiver_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_receiver_t *abcdk_receiver_alloc(const char *tempdir)
{
    abcdk_receiver_t *ctx = NULL;
    
    ctx = abcdk_heap_alloc(sizeof(abcdk_receiver_t));
    if (!ctx)
        return NULL;

    ctx->refcount = 1;
    ctx->offset = 0;
    ctx->tmp_obj = NULL;
    ctx->size = 0;
    ctx->capacity = 0;
    ctx->buf = NULL;

    if (tempdir && *tempdir)
    {
        if (access(tempdir, W_OK) != 0)
            goto final_error;

        strncpy(ctx->tmp_file, tempdir, PATH_MAX - 64);
        abcdk_dirdir(ctx->tmp_file, "abcdk-receiver-XXXXXX");

        ctx->tmp_obj = abcdk_mmap_tempfile(ctx->tmp_file, 4096, 1, 1);
        if (!ctx->tmp_obj)
            goto final_error;
    }

    return ctx;

final_error:

    abcdk_receiver_unref(&ctx);

    return NULL;
}

void *abcdk_receiver_data(const abcdk_receiver_t *ctx)
{
    assert(ctx != NULL);

    return ctx->buf;
}

size_t abcdk_receiver_size(const abcdk_receiver_t *ctx)
{
    assert(ctx != NULL);

    return ctx->size;
}

size_t abcdk_receiver_offset(const abcdk_receiver_t *ctx)
{
    assert(ctx != NULL);

    return ctx->offset;
}

int abcdk_receiver_resize(abcdk_receiver_t *ctx, size_t size)
{
    void *new_buf = NULL;
    int chk;

    assert(ctx != NULL && size > 0);

    /*新的大小与旧的大小一样时，不需要调整。*/
    if (ctx->size == size)
        goto final;

    ctx->size = size;

    /*新的容量与旧的容量一样时，不需要调整。*/
    if (ctx->capacity == abcdk_align(ctx->size + 1, 4096))
        goto final;

    ctx->capacity = abcdk_align(ctx->size + 1, 4096);

    if (ctx->tmp_obj)
    {
        /*内存数据落盘。*/
        chk = abcdk_msync(ctx->tmp_obj,0);
        if (chk != 0)
            return -1;

        /*重新映射文件。*/
        chk = abcdk_mremap(ctx->tmp_obj, ctx->capacity, 1, 1);
        if (chk != 0)
            return -1;

        /*绑定新内存。*/
        ctx->buf = ctx->tmp_obj->pptrs[0];
    }
    else
    {
        /*重新申请内存。*/
        new_buf = abcdk_heap_realloc(ctx->buf, ctx->capacity);
        if (!new_buf)
            return -1;

        /*绑定新内存。*/
        ctx->buf = new_buf;
    }

    // /*多出的一个字节赋值为0。*/
    // ABCDK_PTR2U8(ctx->buf, ctx->capacity - 1) = 0;

final:

    /*修正编移量。*/
    if (ctx->offset > ctx->size)
        ctx->offset = ctx->size;

    return 0;
}

void abcdk_receiver_protocol_set(abcdk_receiver_t *ctx, abcdk_receiver_protocol_t *prot)
{
    assert(ctx != NULL && prot != NULL);
    ABCDK_ASSERT(prot->unpack_cb != NULL,"未绑定解包回调函数，消息对象无法正常工作。");

    ctx->protocol = *prot;
}

int abcdk_receiver_append(abcdk_receiver_t *ctx, const void *data,size_t size,size_t *remain)
{
    ssize_t rsize = 0;
    size_t rall = 0,diff = 0;
    int chk;

    assert(ctx != NULL && data != NULL && size > 0 && remain != NULL);
    
    /*默认无剩余数据。*/
    *remain = 0;

    for (;;)
    {
        /*检测接收的数据是否完整。*/
        if (ctx->protocol.unpack_cb)
        {
            diff = 0;
            chk = ctx->protocol.unpack_cb(ctx->protocol.opaque,ctx->buf,ctx->offset, &diff);
        }
        else
        {
            diff = size - rall;
            chk = 0;
        }

        if (chk != 0)
            break;

        /*检查可用空间。*/
        if (ctx->size - ctx->offset < diff)
        {
            if (abcdk_receiver_resize(ctx, ctx->size + diff) != 0)
                chk = -1;
        }

        if (chk != 0)
            break;

        rsize = ABCDK_MIN(ctx->size - ctx->offset, size - rall);
        rsize = ABCDK_MIN(rsize,diff);
        if (rsize <= 0)
        {
            /*如果未指定解包回调函数，则未知的流数据已经接收完整。*/
            chk = (ctx->protocol.unpack_cb ? 0 : 1);
            break;
        }
        else
        {
            memcpy(ABCDK_PTR2VPTR(ctx->buf, ctx->offset), ABCDK_PTR2VPTR(data, rall), rsize);
            ctx->offset += rsize;
            rall += rsize;
        }
    }

    /*计算剩于数据长度。*/
    *remain = size - rall;

    return chk;
}
