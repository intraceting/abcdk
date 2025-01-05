/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
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