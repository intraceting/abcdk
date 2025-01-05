/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_HEAP_H
#define ABCDK_UTIL_HEAP_H

#include "abcdk/util/defs.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/time.h"

__BEGIN_DECLS

/**
 * 内存回收。
 */
void abcdk_heap_trim (size_t pad);

/**
 * 内存申请(对齐)。
 */
void* abcdk_heap_alloc_align(size_t alignment,size_t size);

/**
 * 内存申请。
 */
void* abcdk_heap_alloc(size_t size);

/**
 * 内存重新申请。
 */
void* abcdk_heap_realloc(void *buf,size_t size);

/**
 * 内存释放。
 * 
 * @param data 内存的指针。
 */
void abcdk_heap_free(void *data);

/**
 * 内存释放。
 * 
 * @param data 指针的指针。返回时赋值NULL(0)。
 */
void abcdk_heap_freep(void **data);

/**
 * 内存克隆。
 * 
 * @note 申请内存大小为size+1。
*/
void *abcdk_heap_clone(const void *data, size_t size);

__END_DECLS

#endif //ABCDK_UTIL_HEAP_H