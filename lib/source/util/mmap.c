/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/mmap.h"

void _abcdk_munmap(abcdk_object_t *obj)
{
    void *mmptr = MAP_FAILED;
    size_t mmsize = 0;
    int fd;

    if (!obj)
        return;

    mmptr = obj->pptrs[1];
    mmsize = obj->sizes[1];
    fd = (long)obj->pptrs[2];

    if (mmptr != MAP_FAILED && mmsize > 0)
        munmap(mmptr, mmsize);
    
    if (fd >= 0)
        abcdk_closep(&fd);

    obj->pptrs[0] = obj->pptrs[1] = obj->pptrs[2] = MAP_FAILED;
}

void _abcdk_munmap_cb(abcdk_object_t *obj, void *opaque)
{
    _abcdk_munmap(obj);
}

abcdk_object_t *_abcdk_mmap(int fd, size_t truncate, int rw, int shared)
{
    abcdk_object_t *obj = NULL;
    void *mmptr = MAP_FAILED;
    int prot = PROT_READ, flags = MAP_PRIVATE;
    struct stat attr;
    int chk;

    obj = abcdk_object_alloc(NULL, 3, 0);
    if (!obj)
        return NULL;

    /*初始化。*/
    obj->pptrs[0] = obj->pptrs[1] = obj->pptrs[2] = MAP_FAILED;

    /* 注册析构函数。 */
    abcdk_object_atfree(obj, _abcdk_munmap_cb, NULL);

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

    /*绑定内存和文件句柄。*/
    obj->pptrs[0] = mmptr;
    obj->sizes[0] = attr.st_size;
    obj->pptrs[1] = mmptr;
    obj->sizes[1] = attr.st_size;
    obj->pptrs[2] = (uint8_t *)(long)dup(fd);

    return obj;

final_error:

    abcdk_object_unref(&obj);

    return NULL;
}

int _abcdk_mremap(abcdk_object_t *obj, size_t truncate, int rw, int shared)
{
    void *mmptr = MAP_FAILED;
    int prot = PROT_READ, flags = MAP_PRIVATE;
    struct stat attr;
    int fd;
    int chk;

    /*复制FD。*/
    fd = dup((long)obj->pptrs[2]);

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

    /*释放旧的内存。*/
    _abcdk_munmap(obj);

    /*绑定内存和文件句柄。*/
    obj->pptrs[0] = mmptr;
    obj->sizes[0] = attr.st_size;
    obj->pptrs[1] = mmptr;
    obj->sizes[1] = attr.st_size;
    obj->pptrs[2] = (uint8_t *)(long)fd;

    return 0;

final_error:

    abcdk_closep(&fd);

    return -1;
}

int abcdk_msync(abcdk_object_t *obj, int async)
{
    int flags;

    assert(obj);
    assert(obj->pptrs[1] != NULL && obj->pptrs[1] != MAP_FAILED && obj->sizes[1] > 0);

    flags = (async ? MS_ASYNC : MS_SYNC);

    return msync(obj->pptrs[1], obj->sizes[1], flags);
}

abcdk_object_t *abcdk_mmap_fd(int fd, size_t truncate, int rw, int shared)
{
    assert(fd >= 0);

    return _abcdk_mmap(fd, truncate, rw, shared);
}

abcdk_object_t *abcdk_mmap_filename(const char *name, size_t truncate, int rw, int shared,int create)
{
    abcdk_object_t *obj = NULL;
    int fd = -1;
    int chk;

    assert(name);

    fd = abcdk_open(name, rw, 0, create);
    if (fd < 0)
        return NULL;

    obj = abcdk_mmap_fd(fd, truncate, rw, shared);
    abcdk_closep(&fd);

    return obj;
}

abcdk_object_t *abcdk_mmap_tempfile(char *name, size_t truncate, int rw, int shared)
{
    abcdk_object_t *obj = NULL;
    int fd = -1;

    assert(name);

    fd = mkstemp(name);
    if (fd < 0)
        return NULL;

    obj = abcdk_mmap_fd(fd, truncate, rw, shared);
    abcdk_closep(&fd);

    return obj;
}

#if !defined(__ANDROID__)

abcdk_object_t* abcdk_mmap_shm(const char* name,size_t truncate,int rw,int shared,int create)
{
    abcdk_object_t *obj = NULL;
    int fd = -1;
    int chk;

    assert(name);

    fd = abcdk_shm_open(name, rw, create);
    if (fd < 0)
        return NULL;

    obj = abcdk_mmap_fd(fd, truncate, rw, shared);
    abcdk_closep(&fd);

    return obj;
}

#endif //__ANDROID__

int abcdk_mremap(abcdk_object_t *obj, size_t truncate, int rw, int shared)
{
    assert(obj);

    return _abcdk_mremap(obj, truncate, rw, shared);
}
