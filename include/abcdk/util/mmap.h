/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_MMAN_H
#define ABCDK_UTIL_MMAN_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"

__BEGIN_DECLS

/**
 * 映射已经打开文件句柄到内存页面。
 *
 * @return NULL(0) 失败，!NULL(0) 成功。
 *
*/
abcdk_object_t* abcdk_mmap(int fd,int rw,int shared);

/**
 * 映射文件到内存页面。
 * 
 * @param name 文件名(或全路径)的指针。
 * 
 * @return NULL(0) 失败，!NULL(0) 成功。
 *  
*/
abcdk_object_t* abcdk_mmap2(const char* name,int rw,int shared);

/**
 * 映射文件到内存页面。
 * 
 * @param [in] truncate 截断文件(或扩展文件)。0 忽略。
 * 
 * @return NULL(0) 失败，!NULL(0) 成功。
*/
abcdk_object_t* abcdk_mmap3(const char* name,size_t truncate,int rw,int shared);

/**
 * 刷新数据。
 * 
 * 如果映射的内存页面是私有模式，对数据修改不会影响原文件。
 * 
 * @param async 0 同步，!0 异步。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_msync(abcdk_object_t* alloc,int async);

/**
 * 解除内存页面映射。
 * 
 * 当前是最后一个引用者才会释放。
 * 
 * @param alloc 指针的指针。当接口返回时，被赋值NULL(0)。
 * 
*/
void abcdk_munmap(abcdk_object_t** alloc);

__END_DECLS


#endif //ABCDK_UTIL_MMAN_H