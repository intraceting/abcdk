/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/endian.h"


int abcdk_endian_check(int big)
{
    long test = 1;

    if (big)
        return (*((char *)&test) != 1);

    return (*((char *)&test) == 1);
}

uint8_t *abcdk_endian_swap(uint8_t *dst, int len)
{
    assert(dst);

    if (len == 2 || len == 3)
    {
        ABCDK_INTEGER_SWAP(dst[0], dst[len - 1]);
    }
    else if (len == 4)
    {
        ABCDK_INTEGER_SWAP(dst[0], dst[3]);
        ABCDK_INTEGER_SWAP(dst[1], dst[2]);
    }
    else if (len == 8)
    {
        ABCDK_INTEGER_SWAP(dst[0], dst[7]);
        ABCDK_INTEGER_SWAP(dst[1], dst[6]);
        ABCDK_INTEGER_SWAP(dst[2], dst[5]);
        ABCDK_INTEGER_SWAP(dst[3], dst[4]);
    }
    else if( len > 1 )
    {
        /* 5,6,7,other,... */
        for (int i = 0; i < len / 2; i++)
            ABCDK_INTEGER_SWAP(dst[len - i - 1], dst[i]);
    }

    return dst;
}

uint8_t* abcdk_endian_b_to_h(uint8_t* dst,int len)
{
    if(abcdk_endian_check(0))
        return abcdk_endian_swap(dst,len);
    
    return dst;
}

uint16_t abcdk_endian_b_to_h16(uint16_t src)
{
    return *((uint16_t*)abcdk_endian_b_to_h((uint8_t*)&src,sizeof(src)));
}

uint32_t abcdk_endian_b_to_h24(const uint8_t* src)
{
    uint32_t dst = 0;

    dst |= src[2];
    dst |= (((uint32_t)src[1]) << 8);
    dst |= (((uint32_t)src[0]) << 16);

    return dst;
}

uint32_t abcdk_endian_b_to_h32(uint32_t src)
{
    return *((uint32_t*)abcdk_endian_b_to_h((uint8_t*)&src,sizeof(src)));
}

uint64_t abcdk_endian_b_to_h64(uint64_t src)
{
    return *((uint64_t*)abcdk_endian_b_to_h((uint8_t*)&src,sizeof(src)));
}

uint8_t* abcdk_endian_h_to_b(uint8_t* dst,int len)
{
    if (abcdk_endian_check(0))
        return abcdk_endian_swap(dst,len);

    return dst;
}

uint16_t abcdk_endian_h_to_b16(uint16_t src)
{
    return *((uint16_t *)abcdk_endian_h_to_b((uint8_t *)&src, sizeof(src)));
}

uint8_t* abcdk_endian_h_to_b24(uint8_t* dst,uint32_t src)
{
    dst[2] = (src & 0xFF);
    dst[1] = ((src >> 8) & 0xFF);
    dst[0] = ((src >> 16) & 0xFF);

    return dst;
}

uint32_t abcdk_endian_h_to_b32(uint32_t src)
{
    return *((uint32_t *)abcdk_endian_h_to_b((uint8_t *)&src, sizeof(src)));
}

uint64_t abcdk_endian_h_to_b64(uint64_t src)
{
    return *((uint64_t *)abcdk_endian_h_to_b((uint8_t *)&src, sizeof(src)));
}

uint8_t* abcdk_endian_l_to_h(uint8_t* dst,int len)
{
    if (abcdk_endian_check(1))
        return abcdk_endian_swap(dst,len);

    return dst;
}

uint16_t abcdk_endian_l_to_h16(uint16_t src)
{
    return *((uint16_t *)abcdk_endian_l_to_h((uint8_t *)&src, sizeof(src)));
}

uint32_t abcdk_endian_l_to_h24(uint8_t* src)
{
    uint32_t dst = 0;

    dst |= src[0];
    dst |= (((uint32_t)src[1]) << 8);
    dst |= (((uint32_t)src[2]) << 16);

    return dst;
}

uint32_t abcdk_endian_l_to_h32(uint32_t src)
{
    return *((uint32_t *)abcdk_endian_l_to_h((uint8_t *)&src, sizeof(src)));
}

uint64_t abcdk_endian_l_to_h64(uint64_t src)
{
    return *((uint64_t *)abcdk_endian_l_to_h((uint8_t *)&src, sizeof(src)));
}

uint8_t* abcdk_endian_h_to_l(uint8_t* dst,int len)
{
    if (abcdk_endian_check(1))
        return abcdk_endian_swap(dst,len);

    return dst;
}

uint16_t abcdk_endian_h_to_l16(uint16_t src)
{
    return *((uint16_t *)abcdk_endian_h_to_l((uint8_t *)&src, sizeof(src)));
}

uint8_t* abcdk_endian_h_to_l24(uint8_t* dst,uint32_t src)
{
    dst[0] = (src & 0xFF);
    dst[1] = ((src >> 8) & 0xFF);
    dst[2] = ((src >> 16) & 0xFF);

    return dst;
}

uint32_t abcdk_endian_h_to_l32(uint32_t src)
{
    return *((uint32_t *)abcdk_endian_h_to_l((uint8_t *)&src, sizeof(src)));
}

uint64_t abcdk_endian_h_to_l64(uint64_t src)
{
    return *((uint64_t *)abcdk_endian_h_to_l((uint8_t *)&src, sizeof(src)));
}
