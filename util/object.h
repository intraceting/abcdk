/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_OBJECT_H
#define ABCDK_UTIL_OBJECT_H

#include "util/general.h"

__BEGIN_DECLS

/**
 * 带引用计数器的内存块信息。
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
     * @warning 如果此项值被调用者覆盖，则需要调用者主动释放，或注册析构函数处理。
     */
    size_t *sizes;

    /**
     * 存放内存块指针的指针数组。
     * 
     * @warning 如果此项值被调用者覆盖，则需要调用者主动释放，或注册析构函数处理。
     */
    uint8_t **pptrs;

} abcdk_object_t;

/**
 * 注册内存块析构函数。
 *
 * @param opaque  环境指针。
*/
void abcdk_object_atfree(abcdk_object_t *alloc,
                           void (*destroy_cb)(abcdk_object_t *alloc, void *opaque),
                           void *opaque);

/**
 * 申请多个内存块。
 * 
 * @param sizes 每个内存块的大小。NULL(0) 容量为0。
 * @param numbers 数量。> 0 的整数。
 * @param drag 拖拽式申请。0 忽略，!0 复制sizes[0]的大小。
*/
abcdk_object_t *abcdk_object_alloc(size_t *sizes, size_t numbers, int drag);

/**
 * 申请一个内存块。
 * 
 * @param size 内存块的大小。>= 0 的整数。
*/
abcdk_object_t *abcdk_object_alloc2(size_t size);

/**
 * 申请多个内存块(数组)。
 * 
 * @param size 内存块的大小。>= 0 的整数。
 * @param numbers 数量。> 0 的整数。
*/
abcdk_object_t *abcdk_object_alloc3(size_t size,size_t numbers);

/**
 * 内存块引用。
 * 
 * 复制内存块指针。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_object_t *abcdk_object_refer(abcdk_object_t *src);

/**
 * 内存块释放。
 * 
 * 当前是最后一个引用者才会释放。
 * 
 * @param [in out] dst 指针的指针。函数返回前设置为NULL(0)。
*/
void abcdk_object_unref(abcdk_object_t **dst);

/**
 * 内存块克隆。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_object_t *abcdk_object_clone(abcdk_object_t *src);

/**
 * 内存块私有化。
 * 
 * 当前是唯一引用者时，直接返回。
 * 当前不是唯一引用者时，执行克隆，然后执行释放。
 *
 * @param [in out] dst 内存块指针的指针。私有化成功后，指针不可再被访问；如果私有化失败，指针依然有效。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_object_t * abcdk_object_privatize(abcdk_object_t **dst);


__END_DECLS

#endif //ABCDK_UTIL_OBJECT_H