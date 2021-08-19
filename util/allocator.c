/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/allocator.h"

/**
 * 带引用计数器的内存块头部。
 * 
 * 申请内存块时，头部和数据块一次性申请创建。
 * 释放内存块时，直接通过头部首地址一次性释放。
*/
typedef struct _abcdk_allocator_hdr
{
    /**
     * 魔法数。
    */
    uint32_t magic;

    /**
     * 'ABCDK' ASCII
    */
#define ABCDK_ALLOCATOR_MAGIC 0x474F4F44

    /**
    * 引用计数器。
    */
    volatile int refcount;

    /**
    * 析构函数。
    */
    void (*destroy_cb)(abcdk_allocator_t *ptr, void *opaque);

    /**
    * 环境指针。
    */
    void *opaque;

    /**
     * 内存块信息。
     * 
     * @note 必须在头部的最后一个元素。
    */
    abcdk_allocator_t out;

} abcdk_allocator_hdr;

/*外部指针转内部指针*/
#define ABCDK_ALLOCATOR_PTR_OUT2IN(PTR) \
    ABCDK_PTR2PTR(abcdk_allocator_hdr, (PTR), -(sizeof(abcdk_allocator_hdr) - sizeof(abcdk_allocator_t)))

/*内部指针转外部指针*/
#define ABCDK_ALLOCATOR_PTR_IN2OUT(PTR) \
    ABCDK_PTR2PTR(abcdk_allocator_t, (PTR), sizeof(abcdk_allocator_hdr) - sizeof(abcdk_allocator_t))

void abcdk_allocator_atfree(abcdk_allocator_t *alloc,
                           void (*destroy_cb)(abcdk_allocator_t *alloc, void *opaque),
                           void *opaque)
{
    abcdk_allocator_hdr *in_p = NULL;

    assert(alloc);
    assert(destroy_cb);

    in_p = ABCDK_ALLOCATOR_PTR_OUT2IN(alloc);

    assert(in_p->magic == ABCDK_ALLOCATOR_MAGIC);

    in_p->destroy_cb = destroy_cb;
    in_p->opaque = opaque;
}

abcdk_allocator_t *abcdk_allocator_alloc(size_t *sizes, size_t numbers, int drag)
{
    abcdk_allocator_hdr *in_p = NULL;
    size_t need_size = 0;
    uint8_t *ptr_p = NULL;

    assert(numbers > 0);

    /*
     * 计算基本的空间。
    */
    need_size += sizeof(abcdk_allocator_hdr);
    need_size += numbers * sizeof(size_t);
    need_size += numbers * sizeof(uint8_t *);

    /*
     * 计算每个内存块的空间。
    */
    if (sizes)
    {
        for (size_t i = 0; i < numbers; i++)
            need_size += (drag ? sizes[0] : sizes[i]);
    }

    /*
     * 一次性申请多个内存块，以便减少多次申请内存块时，碎片化内存块导致内存分页利用率低的问题。
    */
    in_p = (abcdk_allocator_hdr *)abcdk_heap_alloc(need_size);

    if (!in_p)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

    in_p->magic = ABCDK_ALLOCATOR_MAGIC;
    in_p->refcount = 1;
    in_p->destroy_cb = NULL;
    in_p->opaque = NULL;

    in_p->out.refcount = &in_p->refcount;
    in_p->out.numbers = numbers;
    in_p->out.sizes = ABCDK_PTR2PTR(size_t, in_p, sizeof(abcdk_allocator_hdr));
    in_p->out.pptrs = ABCDK_PTR2PTR(uint8_t *, in_p->out.sizes, numbers * sizeof(size_t));

    /*
     * 第一块内存地址。
    */
    ptr_p = ABCDK_PTR2PTR(uint8_t, in_p->out.pptrs, numbers * sizeof(uint8_t *));

    /*
     * 分配每个内存块地址。
    */
    if (sizes)
    {
        for (size_t i = 0; i < numbers; i++)
        {
            /*
             * 内存块容量可能为0，需要跳过。
            */
            in_p->out.sizes[i] = (drag ? sizes[0] : sizes[i]);
            if (in_p->out.sizes[i] <= 0) //(drag == 0 && sizes[i] <= 0)
                continue;

            /*
             * 复制内存块地址。
            */
            in_p->out.pptrs[i] = ptr_p;

            /*
            * 下一块内存块地址。
            */
            ptr_p = ABCDK_PTR2PTR(uint8_t, in_p->out.pptrs[i], in_p->out.sizes[i] * sizeof(uint8_t));
        }
    }

    return ABCDK_ALLOCATOR_PTR_IN2OUT(in_p);
}

abcdk_allocator_t *abcdk_allocator_alloc2(size_t size)
{
    return abcdk_allocator_alloc(&size, 1, 0);
}

abcdk_allocator_t *abcdk_allocator_alloc3(size_t size,size_t numbers)
{
    return abcdk_allocator_alloc(&size, numbers, 1);
}

abcdk_allocator_t *abcdk_allocator_refer(abcdk_allocator_t *src)
{
    abcdk_allocator_hdr *in_p = NULL;
    int chk;

    assert(src);

    in_p = ABCDK_ALLOCATOR_PTR_OUT2IN(src);

    assert(in_p->magic == ABCDK_ALLOCATOR_MAGIC);

    chk = abcdk_atomic_fetch_and_add(&in_p->refcount, 1);
    assert(chk > 0);

    return src;
}

void abcdk_allocator_unref(abcdk_allocator_t **dst)
{
    abcdk_allocator_hdr *in_p = NULL;

    if (!dst || !*dst)
        ABCDK_ERRNO_AND_RETURN0(EINVAL);

    in_p = ABCDK_ALLOCATOR_PTR_OUT2IN(*dst);

    assert(in_p->magic == ABCDK_ALLOCATOR_MAGIC);

    if (abcdk_atomic_fetch_and_add(&in_p->refcount, -1) == 1)
    {
        if (in_p->destroy_cb)
            in_p->destroy_cb(&in_p->out, in_p->opaque);

        in_p->magic = ~(ABCDK_ALLOCATOR_MAGIC);
        in_p->destroy_cb = NULL;
        in_p->opaque = NULL;
        in_p->out.refcount = NULL;
        in_p->out.numbers = 0;
        in_p->out.sizes = NULL;
        in_p->out.pptrs = NULL;

        /* 只要释放一次即可全部释放，因为内存是一次性申请的。*/
        abcdk_heap_free2((void **)&in_p);
    }

    /*Set to NULL(0)*/
    *dst = NULL;
}

abcdk_allocator_t *abcdk_allocator_clone(abcdk_allocator_t *src)
{
    abcdk_allocator_t *dst = NULL;

    assert(src);
    assert(src->numbers > 0 && src->pptrs != NULL && src->sizes != NULL);

    dst = abcdk_allocator_alloc(src->sizes, src->numbers,0);
    if (!dst)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

    for (size_t i = 0; i < src->numbers; i++)
        memcpy(dst->pptrs[i], src->pptrs[i], src->sizes[i]);

    return dst;
}

abcdk_allocator_t *abcdk_allocator_privatize(abcdk_allocator_t **dst)
{
    abcdk_allocator_t *new_p = NULL;
    abcdk_allocator_t *dst_p = NULL;

    assert(dst && *dst);

    new_p = dst_p = *dst;

    if (abcdk_atomic_load((int*)dst_p->refcount) > 1)
    {
        /* 当前不是唯一引用，克隆一份。*/
        new_p = abcdk_allocator_clone(dst_p);
        if(new_p)
            abcdk_allocator_unref(dst);
    }
    else
    {
        /*Set to NULL(0)*/
        *dst = NULL;
    }

    return new_p;
}