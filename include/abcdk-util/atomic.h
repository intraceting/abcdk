/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#ifndef ABCDK_UTIL_ATOMIC_H
#define ABCDK_UTIL_ATOMIC_H

#include "abcdk-util/defs.h"

/*
 * 原子操作。
 *　
 * type __sync_fetch_and_add (type *ptr, type value);
 * type __sync_fetch_and_sub (type *ptr, type value);
 * type __sync_fetch_and_or (type *ptr, type value);
 * type __sync_fetch_and_and (type *ptr, type value);
 * type __sync_fetch_and_xor (type *ptr, type value);
 * type __sync_fetch_and_nand (type *ptr, type value);
 * type __sync_add_and_fetch (type *ptr, type value);
 * type __sync_sub_and_fetch (type *ptr, type value);
 * type __sync_or_and_fetch (type *ptr, type value);
 * type __sync_and_and_fetch (type *ptr, type value);
 * type __sync_xor_and_fetch (type *ptr, type value);
 * type __sync_nand_and_fetch (type *ptr, type value);
 * bool __sync_bool_compare_and_swap (type*ptr, type oldval, type newval, ...)
 * type __sync_val_compare_and_swap (type *ptr, type oldval,  type newval, ...)
*/

/**
 * 返回旧值。
*/
#define abcdk_atomic_load(ptr)   __sync_and_and_fetch((ptr), *(ptr))

/**
 * 设置新值。
*/
#define abcdk_atomic_store(ptr, newval)  __sync_val_compare_and_swap((ptr), *(ptr), (newval))

/**
 * 比较两个值。
*/
#define abcdk_atomic_compare(ptr, oldval)    __sync_bool_compare_and_swap((ptr), (oldval), *(ptr))

/**
 * 比较两个值，相同则用新值替换旧值。
*/
#define abcdk_atomic_compare_and_swap(ptr, oldval, newval)     __sync_bool_compare_and_swap((ptr), (oldval), newval)

/**
 * 加法，返回旧值。
*/
#define abcdk_atomic_fetch_and_add(ptr, val)   __sync_fetch_and_add(ptr, val) 

#endif //ABCDK_UTIL_ATOMIC_H