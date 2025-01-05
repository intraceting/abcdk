/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_HASH_H
#define ABCDK_UTIL_HASH_H

#include "abcdk/util/defs.h"

__BEGIN_DECLS

/**
 * BKDR32
 * 
*/
uint32_t abcdk_hash_bkdr(const void* data,size_t size);

/**
 * BKDR64
 * 
*/
uint64_t abcdk_hash_bkdr64(const void* data,size_t size);

__END_DECLS

#endif //ABCDK_UTIL_HASH_H
