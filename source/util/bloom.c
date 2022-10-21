/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/bloom.h"

int abcdk_bloom_mark(uint8_t *pool, size_t size, size_t number)
{
    assert(pool && size > 0 && size * 8 >= number);

    size_t bloom_pos = 7 - (number & 7);
    size_t byte_pos = number >> 3;
    size_t value = 1 << bloom_pos;

    if((pool[byte_pos] & value) != 0)
        ABCDK_ERRNO_AND_RETURN1(EBUSY,1);

    pool[byte_pos] |= value;

    return 0;
}

int abcdk_bloom_unset(uint8_t* pool,size_t size,size_t number)
{
    assert(pool && size > 0 && size * 8 >= number);

    size_t bloom_pos = 7 - (number & 7);
    size_t byte_pos = number >> 3;
    size_t value = 1 << bloom_pos;

    if((pool[byte_pos] & value) == 0)
        ABCDK_ERRNO_AND_RETURN1(EIDRM,1);

    pool[byte_pos] &= (~value);

    return 0;
}

int abcdk_bloom_filter(uint8_t* pool,size_t size,size_t number)
{
    assert(pool && size > 0 && size * 8 >= number);

    size_t bloom_pos = 7 - (number & 7);
    size_t byte_pos = number >> 3;
    size_t value = 1 << bloom_pos;

    if((pool[byte_pos] & value) != 0)
        return 1;

    return 0;
}

void abcdk_bloom_write(uint8_t* pool,size_t size,size_t offset,int val)
{
    if(val)
        abcdk_bloom_mark(pool,size,offset);
    else 
        abcdk_bloom_unset(pool,size,offset);

    /*Clear error number.*/
    errno = 0;
}

int abcdk_bloom_read(uint8_t* pool,size_t size,size_t offset)
{
    return abcdk_bloom_filter(pool,size,offset);
}
