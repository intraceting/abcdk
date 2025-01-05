/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/shm.h"

#if !defined(__ANDROID__)

int abcdk_shm_open(const char* name,int rw, int create)
{
    int flag = O_RDONLY;
    mode_t mode = S_IRWXU | S_IRWXG | S_IRWXO;

    assert(name);

    if (rw)
        flag = O_RDWR;

    if (rw && create)
        flag |= O_CREAT;

    return shm_open(name,flag,mode);
}

int abcdk_shm_unlink(const char* name)
{
    assert(name);

    return shm_unlink(name);
}

#endif //__ANDROID__
