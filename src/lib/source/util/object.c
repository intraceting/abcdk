/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/object.h"

/**
 * 简单的数据对象.
 * 
 * @note 申请内存块时, 头部和数据块一次性申请创建.
 * @note 释放内存块时, 直接通过头部首地址一次性释放.
*/
typedef struct _abcdk_object_hdr
{
    /** 魔法数.*/
    uint32_t magic;
#define ABCDK_OBJECT_MAGIC 123456789

    /** 引用计数器.*/
    volatile int refcount;

    /** 析构函数.*/
    abcdk_object_destructor_cb destructor_cb;

    /** 环境指针.*/
    void *opaque;

    /** mmap环境.*/
    void *mmap_ptr;
    size_t mmap_size;
    int mmap_fd;

    /**
     * 内存块信息.
     * 
     * @note 必须在头部的最后一个元素.
    */
    abcdk_object_t out;

} abcdk_object_hdr_t;

/*外部指针转内部指针*/
#define ABCDK_OBJECT_PTR_OUT2IN(PTR) \
    ABCDK_PTR2PTR(abcdk_object_hdr_t, (PTR), -(sizeof(abcdk_object_hdr_t) - sizeof(abcdk_object_t)))

/*内部指针转外部指针*/
#define ABCDK_OBJECT_PTR_IN2OUT(PTR) \
    ABCDK_PTR2PTR(abcdk_object_t, (PTR), sizeof(abcdk_object_hdr_t) - sizeof(abcdk_object_t))

void abcdk_object_atfree(abcdk_object_t *obj,abcdk_object_destructor_cb cb,void *opaque)
{
    abcdk_object_hdr_t *in_p = NULL;

    assert(obj != NULL && cb != NULL);

    in_p = ABCDK_OBJECT_PTR_OUT2IN(obj);

    assert(in_p->magic == ABCDK_OBJECT_MAGIC);

    in_p->destructor_cb = cb;
    in_p->opaque = opaque;
}

abcdk_object_t *abcdk_object_alloc(size_t *sizes, size_t numbers, int drag)
{
    abcdk_object_hdr_t *in_p = NULL;
    size_t need_size = 0;
    uint8_t *ptr_p = NULL;

    assert(numbers > 0);

    /* 计算基本的空间.*/
    need_size += sizeof(abcdk_object_hdr_t);
    need_size += numbers * sizeof(size_t);
    need_size += numbers * sizeof(uint8_t *);

    /* 计算每个内存块的空间.*/
    if (sizes)
    {
        for (size_t i = 0; i < numbers; i++)
            need_size += (drag ? sizes[0] : sizes[i]);
    }

    /* 一次性申请多个内存块, 以便减少多次申请内存块时, 碎片化内存块导致内存分页利用率低的问题.*/
    in_p = (abcdk_object_hdr_t *)abcdk_heap_alloc(need_size);

    if (!in_p)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

    in_p->magic = ABCDK_OBJECT_MAGIC;
    in_p->refcount = 1;
    in_p->destructor_cb = NULL;
    in_p->opaque = NULL;
    in_p->mmap_ptr = MAP_FAILED;
    in_p->mmap_size = 0;
    in_p->mmap_fd = -1;

    /* 填充各项信息.*/
    in_p->out.refcount = &in_p->refcount;
    in_p->out.numbers = numbers;
    in_p->out.sizes = ABCDK_PTR2PTR(size_t, in_p, sizeof(abcdk_object_hdr_t));
    in_p->out.pptrs = ABCDK_PTR2PTR(uint8_t *, in_p->out.sizes, numbers * sizeof(size_t));
    in_p->out.pstrs = (char**)in_p->out.pptrs;//copy pointer to `char**`.

    /* 第一块内存地址.*/
    ptr_p = ABCDK_PTR2PTR(uint8_t, in_p->out.pptrs, numbers * sizeof(uint8_t *));

    /* 分配每个内存块地址.*/
    if (sizes)
    {
        for (size_t i = 0; i < numbers; i++)
        {
            /* 内存块容量可能为0, 需要跳过.*/
            in_p->out.sizes[i] = (drag ? sizes[0] : sizes[i]);
            if (in_p->out.sizes[i] <= 0) //(drag == 0 && sizes[i] <= 0)
                continue;

            /* 复制内存块地址.*/
            in_p->out.pptrs[i] = ptr_p;

            /* 下一块内存块地址.*/
            ptr_p = ABCDK_PTR2PTR(uint8_t, in_p->out.pptrs[i], in_p->out.sizes[i]);
        }
    }

    return ABCDK_OBJECT_PTR_IN2OUT(in_p);
}

abcdk_object_t *abcdk_object_alloc2(size_t size)
{
    return abcdk_object_alloc(&size, 1, 0);
}

abcdk_object_t *abcdk_object_alloc3(size_t size,size_t numbers)
{
    return abcdk_object_alloc(&size, numbers, 1);
}

abcdk_object_t *abcdk_object_refer(abcdk_object_t *src)
{
    abcdk_object_hdr_t *in_p = NULL;
    int chk;

    assert(src);

    in_p = ABCDK_OBJECT_PTR_OUT2IN(src);

    assert(in_p->magic == ABCDK_OBJECT_MAGIC);

    chk = abcdk_atomic_fetch_and_add(&in_p->refcount, 1);
    assert(chk > 0);

    return src;
}

void abcdk_object_unref(abcdk_object_t **dst)
{
    abcdk_object_hdr_t *in_p = NULL;

    if (!dst || !*dst)
        return;

    in_p = ABCDK_OBJECT_PTR_OUT2IN(*dst);
    *dst = NULL;

    assert(in_p->magic == ABCDK_OBJECT_MAGIC);

    if (abcdk_atomic_fetch_and_add(&in_p->refcount, -1) != 1)
        return;

    assert(in_p->refcount == 0);
    
    if (in_p->destructor_cb)
        in_p->destructor_cb(&in_p->out, in_p->opaque);

    if (in_p->mmap_ptr != MAP_FAILED && in_p->mmap_size > 0)
        munmap(in_p->mmap_ptr, in_p->mmap_size);
    
    if (in_p->mmap_fd >= 0)
        abcdk_closep(&in_p->mmap_fd);

    in_p->magic = 0xcccccccc;

    in_p->destructor_cb = NULL;
    in_p->opaque = NULL;

    in_p->mmap_ptr = MAP_FAILED;
    in_p->mmap_size = 0;
    in_p->mmap_fd = -1;

    in_p->out.refcount = NULL;
    in_p->out.numbers = 0;
    in_p->out.sizes = NULL;
    in_p->out.pptrs = NULL;
    in_p->out.pstrs = NULL;

    /* 只要释放一次即可全部释放, 因为内存是一次性申请的.*/
    abcdk_heap_free(in_p);
}

abcdk_object_t *abcdk_object_copyfrom(const void *data, size_t size)
{
    abcdk_object_t *obj;

    assert(data != NULL && size > 0);

    /*多申请一个字节.*/
    obj = abcdk_object_alloc2(size + 1);
    if (!obj)
        return NULL;

    memcpy(obj->pptrs[0], data, size);
    obj->sizes[0] = size;

    return obj;
}

abcdk_object_t *abcdk_object_vprintf(int max, const char *fmt, va_list ap)
{
    abcdk_object_t *obj;
    int chk;

    assert(max > 0 && fmt != NULL);

    obj = abcdk_object_alloc2(max + 1);
    if (!obj)
        return NULL;

    chk = vsnprintf(obj->pstrs[0], max, fmt, ap);
    if (chk <= 0)
        goto final_error;

    /*修正格式化后的数据长度.*/
    obj->sizes[0] = chk;

    return obj;

final_error:

    abcdk_object_unref(&obj);

    return NULL;
}

abcdk_object_t *abcdk_object_printf(int max, const char *fmt, ...)
{
    abcdk_object_t *obj;

    assert(max > 0 && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    obj = abcdk_object_vprintf(max, fmt, ap);
    va_end(ap);

    return obj;
}

abcdk_object_t *abcdk_object_copypair(const void *key, size_t ksize, const void *val, size_t vsize)
{
    abcdk_object_t *obj;

    assert(key != NULL && ksize > 0 && val != NULL && vsize > 0);

    /*多申请一个字节.*/
    ssize_t ssize[] = {ksize + 1, vsize + 1};
    obj = abcdk_object_alloc(ssize, 2, 0);
    if (!obj)
        return NULL;

    memcpy(obj->pptrs[0], key, ksize);
    obj->sizes[0] = ksize;

    memcpy(obj->pptrs[1], val, vsize);
    obj->sizes[1] = vsize;

    return obj;
}

abcdk_object_t *abcdk_object_copyfrom_file(const void *file)
{
    int fd = -1;
    struct stat attr = {0};
    abcdk_object_t *obj = NULL;
    ssize_t rlen;

    assert(file != NULL);

    fd = abcdk_open(file,0,0,0);
    if(fd < 0)
        return NULL;

    fstat(fd,&attr);

    if(attr.st_size > 0)
    {
        obj = abcdk_object_alloc2(attr.st_size + 1);//多一个字节.
        if (obj)
        {

            obj->sizes[0] = attr.st_size;//修复长度.

            rlen = abcdk_read(fd, obj->pptrs[0], obj->sizes[0]);
            if (rlen != obj->sizes[0])
                abcdk_object_unref(&obj);
        }
    }

    abcdk_closep(&fd);
    return obj;
}

abcdk_object_t *abcdk_object_mmap(int fd, size_t truncate, int rw, int shared)
{
    abcdk_object_t *obj = NULL;
    abcdk_object_hdr_t *in_p = NULL;
    void *mmptr = MAP_FAILED;
    int prot = PROT_READ, flags = MAP_PRIVATE;
    struct stat attr;
    int chk;

    assert(fd >= 0);

    obj = abcdk_object_alloc(NULL, 1, 0);
    if (!obj)
        return NULL;

    in_p = ABCDK_OBJECT_PTR_OUT2IN(obj);

    if (truncate > 0)
    {
        chk = ftruncate(fd, truncate);
        if (chk != 0)
            goto final_error;
    }

    if (fstat(fd, &attr) == -1)
        goto final_error;

    if (attr.st_size <= 0)
        goto final_error;

    if (rw)
        prot = PROT_READ | PROT_WRITE;
    if (shared)
        flags = MAP_SHARED;

    mmptr = mmap(0, attr.st_size, prot, flags, fd, 0);
    if (mmptr == MAP_FAILED)
        goto final_error;

    /*绑定内存和文件句柄(外部).*/
    obj->pptrs[0] = mmptr;
    obj->sizes[0] = attr.st_size;

    /*绑定内存和文件句柄(内部).*/
    in_p->mmap_ptr = mmptr;
    in_p->mmap_size = attr.st_size;
    in_p->mmap_fd = dup(fd);

    return obj;

final_error:

    abcdk_object_unref(&obj);

    return NULL;
}


abcdk_object_t *abcdk_object_mmap_filename(const char *name, size_t truncate, int rw, int shared,int create)
{
    abcdk_object_t *obj = NULL;
    int fd = -1;
    int chk;

    assert(name);

    fd = abcdk_open(name, rw, 0, create);
    if (fd < 0)
        return NULL;

    obj = abcdk_object_mmap(fd, truncate, rw, shared);
    abcdk_closep(&fd);

    return obj;
}

abcdk_object_t *abcdk_object_mmap_tempfile(char *name, size_t truncate, int rw, int shared)
{
    abcdk_object_t *obj = NULL;
    int fd = -1;

    assert(name);

    fd = mkstemp(name);
    if (fd < 0)
        return NULL;

    obj = abcdk_object_mmap(fd, truncate, rw, shared);
    abcdk_closep(&fd);

    return obj;
}

#if !defined(__ANDROID__)

abcdk_object_t* abcdk_object_mmap_shm(const char* name,size_t truncate,int rw,int shared,int create)
{
    abcdk_object_t *obj = NULL;
    int fd = -1;
    int chk;

    assert(name);

    fd = abcdk_shm_open(name, rw, create);
    if (fd < 0)
        return NULL;

    obj = abcdk_object_mmap(fd, truncate, rw, shared);
    abcdk_closep(&fd);

    return obj;
}

#endif //__ANDROID__

int abcdk_object_remmap(abcdk_object_t *obj, size_t truncate, int rw, int shared)
{
    abcdk_object_hdr_t *in_p = NULL;
    void *mmptr = MAP_FAILED;
    int prot = PROT_READ, flags = MAP_PRIVATE;
    struct stat attr;
    int fd;
    int chk;

    assert(obj != NULL);

    in_p = ABCDK_OBJECT_PTR_OUT2IN(obj);

    ABCDK_TRACE_ASSERT(in_p->mmap_fd >= 0, ABCDK_GETTEXT("内存对象不支持此项操作."));

    if (truncate > 0)
    {
        chk = ftruncate(in_p->mmap_fd, truncate);
        if (chk != 0)
            return -1;
    }

    if (fstat(in_p->mmap_fd, &attr) == -1)
        return -2;

    if (attr.st_size <= 0)
        return -3;

    if (rw)
        prot = PROT_READ | PROT_WRITE;
    if (shared)
        flags = MAP_SHARED;

    mmptr = mmap(0, attr.st_size, prot, flags, in_p->mmap_fd, 0);
    if (mmptr == MAP_FAILED)
        return -4;

    /*释放旧的内存映射.*/
    if (in_p->mmap_ptr != MAP_FAILED && in_p->mmap_size > 0)
        munmap(in_p->mmap_ptr, in_p->mmap_size);

    /*重新绑定内存和文件句柄(外部).*/
    obj->pptrs[0] = mmptr;
    obj->sizes[0] = attr.st_size;

    /*重新绑定内存和文件句柄(内部).*/
    in_p->mmap_ptr = mmptr;
    in_p->mmap_size = attr.st_size;

    return 0;
}

int abcdk_object_msync(abcdk_object_t *obj, int async)
{
    abcdk_object_hdr_t *in_p = NULL;
    int flags;
    int chk;

    assert(obj);

    in_p = ABCDK_OBJECT_PTR_OUT2IN(obj);

    ABCDK_TRACE_ASSERT(in_p->mmap_fd >= 0, ABCDK_GETTEXT("内存对象不支持此项操作."));

    flags = (async ? MS_ASYNC : MS_SYNC);

    chk = msync(in_p->mmap_ptr, in_p->mmap_size, flags);
    if (chk != 0)
        return -1;

    return 0;
}
