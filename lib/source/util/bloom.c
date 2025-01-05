/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/bloom.h"

int abcdk_bloom_mark(uint8_t *pool, size_t size, size_t index)
{
    assert(pool != NULL && size > 0);
    assert(index < size * 8);

    size_t bloom_pos = 7 - (index & 7);
    size_t byte_pos = index >> 3;
    size_t value = 1 << bloom_pos;

    if((pool[byte_pos] & value) != 0)
        ABCDK_ERRNO_AND_RETURN1(EBUSY,1);

    pool[byte_pos] |= value;

    return 0;
}

int abcdk_bloom_unset(uint8_t* pool,size_t size,size_t index)
{
    assert(pool != NULL && size > 0);
    assert(index < size * 8);

    size_t bloom_pos = 7 - (index & 7);
    size_t byte_pos = index >> 3;
    size_t value = 1 << bloom_pos;

    if((pool[byte_pos] & value) == 0)
        ABCDK_ERRNO_AND_RETURN1(EIDRM,1);

    pool[byte_pos] &= (~value);

    return 0;
}

int abcdk_bloom_filter(const uint8_t* pool,size_t size,size_t index)
{
    assert(pool != NULL && size > 0);
    assert(index < size * 8);

    size_t bloom_pos = 7 - (index & 7);
    size_t byte_pos = index >> 3;
    size_t value = 1 << bloom_pos;

    if((pool[byte_pos] & value) != 0)
        return 1;

    return 0;
}

void abcdk_bloom_write(uint8_t* pool,size_t size,size_t offset,int val)
{
    assert(pool != NULL && size > 0);
    assert(offset < size * 8);

    if(val)
        abcdk_bloom_mark(pool,size,offset);
    else 
        abcdk_bloom_unset(pool,size,offset);

    /*Clear error number.*/
    errno = 0;
}

int abcdk_bloom_read(const uint8_t* pool,size_t size,size_t offset)
{
    assert(pool != NULL && size > 0);
    assert(offset < size * 8);

    return abcdk_bloom_filter(pool,size,offset);
}

uint64_t abcdk_bloom_read_number(const uint8_t *pool, size_t size, size_t offset, int bits)
{
    uint64_t num = 0;

    assert(pool != NULL && size > 0 && bits > 0);
    assert(offset + bits <= size * 8);

    if (offset % 8 == 0 && bits % 8 == 0)
    {
        for (int i = 0; i < bits / 8; i++)
        {
            num = (num << 8) | pool[offset / 8];
            offset += 8;
        }
    }
    else
    {
        for (int i = 0; i < bits; i++)
            num = (num << 1) | abcdk_bloom_read(pool, size, offset + i);
    }

    return num;
}

void abcdk_bloom_write_number(uint8_t *pool, size_t size, size_t offset, int bits, uint64_t num)
{
    assert(pool != NULL && size > 0 && bits > 0);
    assert(offset + bits <= size * 8);

    if (offset % 8 == 0 && bits % 8 == 0)
    {
        for (int i = (bits / 8 - 1); i >= 0; i--)
        {
            pool[offset / 8] = ((num >> i * 8) & 0xff);
            offset += 8;
        }
    }
    else
    {
        for (int i = 0; i < bits; i++)
            abcdk_bloom_write(pool, size, offset + i, ((num >> (bits - 1 - i)) & 1));
    }
}