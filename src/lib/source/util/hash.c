/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */

#include "abcdk/util/hash.h"

uint32_t abcdk_hash_bkdr(const void* data,size_t size)
{
    uint32_t seed = 131; /* 31 131 1313 13131 131313 etc.. */
    uint32_t hash = 0;

    assert(data && size>0);

    for (size_t i = 0; i < size;i++)
    {
        hash = (hash * seed) + ABCDK_PTR2OBJ(uint8_t,data,i);
    }

    return hash;
}

uint64_t abcdk_hash_bkdr64(const void* data,size_t size)
{
    uint64_t seed = 13113131; /* 31 131 1313 13131 131313 etc.. */
    uint64_t hash = 0;

    assert(data && size>0);

    for (size_t i = 0; i < size;i++)
    {
        hash = (hash * seed) + ABCDK_PTR2OBJ(uint8_t,data,i);
    }

    return hash; 
}

uint64_t abcdk_hash_better64(const void* data,size_t size)
{
    //1: 用现有的 BKDR 快速处理.
    uint64_t h = abcdk_hash_bkdr64(data, size);

    //2: 叠加 MurmurHash3 的收尾混合逻辑(fmix64) 使变化更加明显.
    h ^= (h >> 33);
    h *= 0xff51afd7ed558ccdLLU;
    h ^= (h >> 33);
    h *= 0xc4ceb9fe1a85ec53LLU;
    h ^= (h >> 33);

    return h;
}