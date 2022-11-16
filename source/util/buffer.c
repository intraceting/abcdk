/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/util/buffer.h"

abcdk_buffer_t *abcdk_buffer_alloc(abcdk_object_t *obj)
{
    abcdk_buffer_t *buf = NULL;

    buf = abcdk_heap_alloc(sizeof(abcdk_buffer_t));
    if (!buf)
        return NULL;

    if (obj)
    {
        assert(obj->numbers > 0 && obj->pptrs[0] != NULL && obj->sizes[0] > 0);

        /*绑定内存块。*/
        buf->obj = obj;

        buf->data = buf->obj->pptrs[0];
        buf->size = buf->obj->sizes[0];

        buf->rsize = buf->wsize = 0;
    }
    else
    {
        /*允许空的。*/
        buf->obj = buf->data = NULL;
        buf->size = buf->rsize = buf->wsize = 0;
    }

    return buf;
}

abcdk_buffer_t *abcdk_buffer_alloc2(size_t size)
{
    abcdk_buffer_t *buf = NULL;
    abcdk_object_t *alloc = NULL;

    if(size > 0)
    {
        alloc = abcdk_object_alloc2(size);
        if (!alloc)
            goto final_error;
    }

    buf = abcdk_buffer_alloc(alloc);
    if (!buf)
        goto final_error;

    return buf;

final_error:

    abcdk_object_unref(&alloc);

    return NULL;
}

void abcdk_buffer_free(abcdk_buffer_t **dst)
{
    abcdk_buffer_t *buf_p = NULL;

    if (!dst || !*dst)
        ABCDK_ERRNO_AND_RETURN0(EINVAL);

    buf_p = *dst;

    abcdk_object_unref(&buf_p->obj);

    abcdk_heap_free2((void **)dst);
}

int abcdk_buffer_resize(abcdk_buffer_t *buf, size_t size)
{
    abcdk_object_t *alloc_new = NULL;

    assert(buf != NULL && size > 0);

    if (buf->size == size)
        return 0;

    alloc_new = abcdk_object_alloc2(size);
    if (!alloc_new)
        return -1;

    /*复制数据。*/
    memcpy(alloc_new->pptrs[0], buf->data, buf->size);

    /*解除旧的内存块*/
    abcdk_object_unref(&buf->obj);

    /*绑定新的内存块。*/
    buf->obj = alloc_new;

    buf->data = alloc_new->pptrs[0];
    buf->size = alloc_new->sizes[0];

    if (buf->wsize > size)
        buf->wsize = size;
    if (buf->rsize > size)
        buf->rsize = size;

    return 0;
}

ssize_t abcdk_buffer_write(abcdk_buffer_t *buf, const void *data, size_t size)
{
    ssize_t wsize2 = 0;

    assert(buf != NULL && data != NULL && size > 0);
    assert(buf->data != NULL && buf->size > 0);

    if (buf->wsize >= buf->size)
        ABCDK_ERRNO_AND_RETURN1(ENOSPC, 0);

    wsize2 = ABCDK_MIN(buf->size - buf->wsize, size);
    memcpy(ABCDK_PTR2PTR(void, buf->data, buf->wsize), data, wsize2);
    buf->wsize += wsize2;

    return wsize2;
}

ssize_t abcdk_buffer_read(abcdk_buffer_t *buf, void *data, size_t size)
{
    ssize_t rsize2 = 0;

    assert(buf != NULL && data != NULL && size > 0);
    assert(buf->data != NULL && buf->size > 0);

    if (buf->rsize >= buf->wsize)
        ABCDK_ERRNO_AND_RETURN1(ESPIPE, 0);

    rsize2 = ABCDK_MIN(buf->wsize - buf->rsize, size);
    memcpy(data, ABCDK_PTR2PTR(void, buf->data, buf->rsize), rsize2);
    buf->rsize += rsize2;

    return rsize2;
}

ssize_t abcdk_buffer_readline(abcdk_buffer_t *buf, void *data, size_t size, int delim)
{
    ssize_t rsize2 = 0;
    ssize_t rsize3 = 0;

    assert(buf != NULL && data != NULL && size > 0);
    assert(buf->data != NULL && buf->size > 0);

    if (buf->rsize >= buf->wsize)
        ABCDK_ERRNO_AND_RETURN1(ESPIPE, 0);

    /*查找行尾标志。*/
    for (size_t i = 0; i < (buf->wsize - buf->rsize); i++)
    {
        rsize3 += 1;
        if (ABCDK_PTR2I8(buf->data, buf->rsize + i) == delim)
            break;
    }

    rsize2 = ABCDK_MIN(rsize3, size);
    memcpy(data, ABCDK_PTR2VPTR(buf->data, buf->rsize), rsize2);
    buf->rsize += rsize3;//累加行真实长度。

    /*添加结束符。*/
    if (rsize3 < size)
        ABCDK_PTR2I8(data, rsize2) = '\0';

    return rsize2;
}

void abcdk_buffer_drain(abcdk_buffer_t *buf)
{
    assert(buf != NULL);
    assert(buf->data != NULL && buf->size > 0);

    assert(buf->rsize <= buf->wsize);

    if (buf->rsize > 0)
    {
        buf->wsize -= buf->rsize;
        memmove(buf->data, ABCDK_PTR2PTR(void, buf->data, buf->rsize), buf->wsize);
        buf->rsize = 0;
    }
}

ssize_t abcdk_buffer_fill(abcdk_buffer_t *buf, uint8_t stuffing)
{
    ssize_t wsize2 = 0;

    assert(buf != NULL);
    assert(buf->data != NULL && buf->size > 0);

    if (buf->wsize >= buf->size)
        ABCDK_ERRNO_AND_RETURN1(ENOSPC, 0);

    wsize2 = buf->size - buf->wsize;
    memset(ABCDK_PTR2PTR(void, buf->data, buf->wsize), stuffing, wsize2);
    buf->wsize += wsize2;

    return wsize2;
}

ssize_t abcdk_buffer_vprintf(abcdk_buffer_t *buf, const char *fmt, va_list args)
{
    ssize_t wsize2 = 0;

    assert(buf != NULL && fmt != NULL);
    assert(buf->data != NULL && buf->size > 0);

    if (buf->wsize >= buf->size)
        ABCDK_ERRNO_AND_RETURN1(ENOSPC, 0);

    wsize2 = vsnprintf(ABCDK_PTR2PTR(char, buf->data, buf->wsize),
                       buf->size - buf->wsize, fmt, args);
    if (wsize2 > 0)
        buf->wsize += wsize2;

    return wsize2;
}

ssize_t abcdk_buffer_printf(abcdk_buffer_t *buf, const char *fmt, ...)
{
    ssize_t wsize2 = 0;

    assert(buf != NULL && fmt != NULL);
    assert(buf->data != NULL && buf->size > 0);

    va_list args;
    va_start(args, fmt);

    wsize2 = abcdk_buffer_vprintf(buf, fmt, args);

    va_end(args);

    return wsize2;
}

ssize_t abcdk_buffer_import(abcdk_buffer_t *buf, int fd)
{
    struct stat attr = {0};

    assert(buf != NULL && fd >= 0);

    if (fstat(fd, &attr) == -1)
        ABCDK_ERRNO_AND_RETURN1(EBADF, -1);

    return abcdk_buffer_import_atmost(buf, fd, attr.st_size);
}

ssize_t abcdk_buffer_import_atmost(abcdk_buffer_t *buf, int fd, size_t howmuch)
{
    ssize_t wsize2 = 0;
    ssize_t wsize3 = 0;

    assert(buf != NULL && fd >= 0 && howmuch > 0);
    assert(buf->data != NULL && buf->size > 0);

    if (buf->wsize >= buf->size)
        ABCDK_ERRNO_AND_RETURN1(ENOSPC, 0);

    wsize2 = ABCDK_MIN(buf->size - buf->wsize, howmuch);
    wsize3 = abcdk_read(fd, ABCDK_PTR2PTR(void, buf->data, buf->wsize), wsize2);
    if (wsize3 > 0)
        buf->wsize += wsize3;

    return wsize3;
}

ssize_t abcdk_buffer_export(abcdk_buffer_t *buf, int fd)
{
    return abcdk_buffer_export_atmost(buf, fd, SIZE_MAX);
}

ssize_t abcdk_buffer_export_atmost(abcdk_buffer_t *buf, int fd, size_t howmuch)
{
    ssize_t rsize2 = 0;
    ssize_t rsize3 = 0;

    assert(buf != NULL && fd >= 0 && howmuch > 0);
    assert(buf->data != NULL && buf->size > 0);

    if (buf->rsize >= buf->wsize)
        ABCDK_ERRNO_AND_RETURN1(ESPIPE, 0);

    rsize2 = ABCDK_MIN(buf->wsize - buf->rsize, howmuch);
    rsize3 = abcdk_write(fd, ABCDK_PTR2PTR(void, buf->data, buf->rsize), rsize2);
    if (rsize3 > 0)
        buf->rsize += rsize3;

    return rsize3;
}