/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/general.h"

size_t abcdk_align(size_t size, size_t align)
{
    size_t pad = 0;

    if (align > 1)
    {
        pad = size % align;
        size += ((pad > 0) ? (align - pad) : 0);
    }

    return size;
}

int abcdk_once(volatile int *status, int (*routine)(void *opaque), void *opaque)
{
    int chk, ret;

    assert(status != NULL && opaque != NULL);

    if (abcdk_atomic_compare_and_swap(status,0, 1))
    {
        ret = 0;
        chk = routine(opaque);
        abcdk_atomic_store(status, ((chk == 0) ? 2 : 0));
    }
    else
    {
        ret = 1;
        while (abcdk_atomic_load(status) == 1)
            sched_yield();
    }

    chk = ((abcdk_atomic_load(status) == 2) ? 0 : -1);

    return (chk == 0 ? ret : -1);
}

char *abcdk_bin2hex(char* dst,const void *src,size_t size, int ABC)
{
    assert(dst != NULL && src != NULL && size>0);

    for (size_t i = 0; i < size; i++)
    {
        if (ABC)
            sprintf(ABCDK_PTR2U8PTR(dst, i * 2),"%02X", ABCDK_PTR2U8(src,i));
        else
            sprintf(ABCDK_PTR2U8PTR(dst, i * 2),"%02x", ABCDK_PTR2U8(src,i));
    }  
    return dst;
}

void *abcdk_hex2bin(void *dst, const char *src, size_t size)
{
    assert(dst != NULL && src != NULL && size > 0);
    assert(size % 2 == 0);

    for (size_t i = 0; i < size / 2; i++)
    {
        sscanf(ABCDK_PTR2U8PTR(src, i * 2), "%2hhx", ABCDK_PTR2U8PTR(dst,i));
    }
  
    return dst;
}

void *abcdk_cyclic_shift(void *data, size_t size, size_t bits, int direction)
{
    uint8_t m, t;

    assert(data != NULL && size > 0);
    assert(direction == 1 || direction == 2);

    if (bits <= 0)
        goto final;

    /* 每个字节8bit，移位超过8bit时需要特殊处理。*/
    for (; bits > 8; bits -= 8)
        abcdk_cyclic_shift(data, size, 8, direction);

    if (direction == 1)
    {
        m = (0xFF << (8 - bits));
        t = (ABCDK_PTR2U8(data, 0) & m);
        for (size_t i = 0; i < size - 1; i++)
        {
            ABCDK_PTR2U8(data, i) <<= bits;
            ABCDK_PTR2U8(data, i) |= (ABCDK_PTR2U8(data, i + 1) >> (8 - bits));
        }
        ABCDK_PTR2U8(data, size - 1) <<= bits;
        ABCDK_PTR2U8(data, size - 1) |= (t >> (8 - bits));
    }
    else /*if (direction == 2)*/
    {
        m = (0xFF >> (8 - bits));
        t = (ABCDK_PTR2U8(data, size - 1) & m);
        for (size_t i = size - 1; i > 0; i--)
        {
            ABCDK_PTR2U8(data, i) >>= bits;
            ABCDK_PTR2U8(data, i) |= (ABCDK_PTR2U8(data, i - 1) << (8 - bits));
        }
        ABCDK_PTR2U8(data, 0) >>= bits;
        ABCDK_PTR2U8(data, 0) |= (t << (8 - bits));
    }

final:

    return data;
}
