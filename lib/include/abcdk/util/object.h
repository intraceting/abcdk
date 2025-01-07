/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_OBJECT_H
#define ABCDK_UTIL_OBJECT_H

#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"

__BEGIN_DECLS

/**
 * 简单的数据对象。
*/
typedef struct _abcdk_object
{
    /**
     * 引用计数器指针。
    */
    const volatile int *refcount;

    /**
     * 内存块数量。
    */
    size_t numbers;

    /**
     * 存放内存块大小的指针数组。
     *
     * @note 如果此项值被调用者覆盖，则需要调用者主动释放，或注册析构函数处理。
     */
    size_t *sizes;

    /**
     * 存放内存块指针的指针数组。
     * 
     * @note 如果此项值被调用者覆盖，则需要调用者主动释放，或注册析构函数处理。
     */
    uint8_t **pptrs;
    char **pstrs;

} abcdk_object_t;

/**
 * 析构函数。
 * 
 * @param [in] opaque 用户环境指针。
 */
typedef void (*abcdk_object_destructor_cb)(abcdk_object_t *obj, void *opaque);

/**
 * 注册析构函数。
 *
 * @param opaque  环境指针。
 */
void abcdk_object_atfree(abcdk_object_t *obj,abcdk_object_destructor_cb cb,void *opaque);

/**
 * 申请。
 * 
 * @param sizes 每个内存块的大小。NULL(0) 容量为0。
 * @param numbers 数量。> 0 的整数。
 * @param drag 拖拽式申请。0 忽略，!0 复制sizes[0]的大小。
*/
abcdk_object_t *abcdk_object_alloc(size_t *sizes, size_t numbers, int drag);

/**
 * 申请。
 * 
 * @param size 内存块的大小。>= 0 的整数。
*/
abcdk_object_t *abcdk_object_alloc2(size_t size);

/**
 * 申请。
 * 
 * @param size 内存块的大小。>= 0 的整数。
 * @param numbers 数量。> 0 的整数。
*/
abcdk_object_t *abcdk_object_alloc3(size_t size,size_t numbers);

/**
 * 引用。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_object_t *abcdk_object_refer(abcdk_object_t *src);

/**
 * 释放。
 * 
 * @param [in out] dst 指针的指针。
*/
void abcdk_object_unref(abcdk_object_t **dst);

/** 申请一个内存块，并复制数据。*/
abcdk_object_t *abcdk_object_copyfrom(const void *data, size_t size);

/** 申请一个内存块，并格式化数据。*/
abcdk_object_t *abcdk_object_vprintf(int max, const char *fmt, va_list ap);

/** 申请一个内存块，并格式化数据。*/
abcdk_object_t *abcdk_object_printf(int max, const char *fmt, ...);

/** 申请一个内存块，并复制数据。*/
abcdk_object_t *abcdk_object_copypair(const void *key, size_t ksize,const void *val, size_t vsize);

__END_DECLS

#endif //ABCDK_UTIL_OBJECT_H