/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_UTIL_ATOMIC_H
#define ABCDK_UTIL_ATOMIC_H

#include "abcdk/util/defs.h"
#include "abcdk/util/mutex.h"

/** 加锁.*/
void abcdk_atomic_lock(void);

/** 解锁.*/
void abcdk_atomic_unlock(void);

/** 返回旧值.*/
#define abcdk_atomic_load(ptr)                 \
    ({                                         \
        __typeof__(*(ptr)) *_p = (ptr);        \
        __atomic_load_n(_p, __ATOMIC_SEQ_CST); \
    })

/** 设置新值.*/
#define abcdk_atomic_store(ptr, newval)              \
    ({                                               \
        __typeof__(*(ptr)) *_p = (ptr);              \
        __typeof__(*(ptr)) _nv = (newval);           \
        __atomic_store_n(_p, _nv, __ATOMIC_SEQ_CST); \
    })

/** 比较两个值, 返回布尔值.*/
#define abcdk_atomic_compare(ptr, oldval)               \
    ({                                                  \
        __typeof__(*(ptr)) *_p = (ptr);                 \
        __typeof__(*(ptr)) _ov = (oldval);              \
        (__atomic_load_n(_p, __ATOMIC_SEQ_CST) == _ov); \
    })

/** 比较两个值, 相同则用新值交换, 返回布尔值.*/
#define abcdk_atomic_compare_and_swap(ptr, expected, newval)                                \
    ({                                                                                      \
        __typeof__(*(ptr)) *_p = (ptr);                                                     \
        __typeof__(*(ptr)) _exp = (expected);                                               \
        __typeof__(*(ptr)) _nv = (newval);                                                  \
        __atomic_compare_exchange_n(_p, &_exp, _nv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); \
    })

/** 加法, 返回旧值.*/
#define abcdk_atomic_fetch_and_add(ptr, val)          \
    ({                                                \
        __typeof__(*(ptr)) *_p = (ptr);               \
        __typeof__(*(ptr)) _v = (val);                \
        __atomic_fetch_add(_p, _v, __ATOMIC_SEQ_CST); \
    })

/** 加法, 返回新值. */
#define abcdk_atomic_add_and_fetch(ptr, val)          \
    ({                                                \
        __typeof__(*(ptr)) *_p = (ptr);               \
        __typeof__(*(ptr)) _v = (val);                \
        __atomic_add_fetch(_p, _v, __ATOMIC_SEQ_CST); \
    })

#endif // ABCDK_UTIL_ATOMIC_H