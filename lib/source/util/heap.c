/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/heap.h"

void *abcdk_heap_alloc(size_t size)
{
    assert(size > 0);

    return calloc(1,size);
}

void* abcdk_heap_realloc(void *buf,size_t size)
{
    assert(size > 0);

    return realloc(buf,size);
}

void abcdk_heap_free(void *data)
{
    if (data)
        free(data);
}

void abcdk_heap_free2(void **data)
{
    void *data_p;

    if (!data || !*data)
        return;

    data_p = *data;
    *data = NULL;

    abcdk_heap_free(data_p);
}

void *abcdk_heap_clone(const void *data, size_t size)
{
    void *buf = NULL;

    assert(data && size > 0);

    buf = abcdk_heap_alloc(size + 1);
    if (!buf)
        ABCDK_ERRNO_AND_RETURN1(EINVAL, NULL);

    memcpy(buf, data, size);

    return buf;
}