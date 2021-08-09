/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_BUFFER_H
#define ABCDK_UTIL_BUFFER_H

#include "abcdk-util/general.h"
#include "abcdk-util/allocator.h"

__BEGIN_DECLS

/**
 * 简单的缓存。
 * 
*/
typedef struct _abcdk_buffer
{
    /**
     * 内存块。
     *
     * @note 尽量不要直接访问。
    */
    abcdk_allocator_t *alloc;

    /**
     * 内存指针。
    */
    void* data;

    /**
     * 容量大小。
    */
    size_t size;

    /**
     * 已读大小。
     * 
    */
    size_t rsize;

    /**
     * 已写大小。
    */
    size_t wsize;

} abcdk_buffer_t;

/**
 * 创建。
 * 
 * @param alloc 内存块的指针。仅复制指针，不是指针对像引用。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
 */
abcdk_buffer_t *abcdk_buffer_alloc(abcdk_allocator_t *alloc);

/**
 * 创建。
 * 
 * @param size 容量(Bytes)。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
 * 
 */
abcdk_buffer_t *abcdk_buffer_alloc2(size_t size);

/**
 * 释放。
 * 
 * @param dst 缓存指针的指针。函数返回前修改为NULL(0);
*/
void abcdk_buffer_free(abcdk_buffer_t **dst);

/**
 * 复制。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
 */
abcdk_buffer_t *abcdk_buffer_copy(abcdk_buffer_t *src);

/**
 * 克隆。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
 */
abcdk_buffer_t *abcdk_buffer_clone(abcdk_buffer_t *src);

/**
 * 私有化。
 * 
 * 用于写前复制，或克隆引用的内存块。如果是非引用内存块，直接返回成功。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_buffer_privatize(abcdk_buffer_t *dst);

/**
 * 调整容量。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_buffer_resize(abcdk_buffer_t *buf,size_t size);

/**
 * 写入数据。
 * 
 * @return > 0 写入的长度(Bytes)，= 0 已满，< 0 出错。
*/
ssize_t abcdk_buffer_write(abcdk_buffer_t *buf, const void *data, size_t size);

/**
 * 读取数据。
 * 
 * @return > 0 读取的长度(Bytes)，= 0 末尾，< 0 出错。
*/
ssize_t abcdk_buffer_read(abcdk_buffer_t *buf, void *data, size_t size);

/**
 * 读取一行数据。
 * 
 * @warning 当缓存不足时，行尾部分将被截断并丢弃。
 * 
 * @param delim 行分割符。
 * 
 * @return > 0 读取的长度(Bytes)，= 0 末尾，< 0 出错。
*/
ssize_t abcdk_buffer_readline(abcdk_buffer_t *buf, void *data, size_t size, int delim);

/**
 * 排出已读数据，未读数据移动到缓存首地址。
*/
void abcdk_buffer_drain(abcdk_buffer_t *buf);

/**
 * 填满缓存。
 * 
 * @param stuffing 填充物。
 * 
 * @return > 0 添加的长度(Bytes)，= 0 已满，< 0 出错。
*/
ssize_t abcdk_buffer_fill(abcdk_buffer_t *buf,uint8_t stuffing);

/**
 * 格式化写入数据。
 * 
 * @return > 0 写入的长度(Bytes)，= 0 已满，< 0 出错。
*/
ssize_t abcdk_buffer_vprintf(abcdk_buffer_t *buf,const char * fmt, va_list args);

/**
 * 格式化写入数据。
 * 
 * @return > 0 写入的长度(Bytes)，= 0 已满，< 0 出错。
*/
ssize_t abcdk_buffer_printf(abcdk_buffer_t *buf,const char * fmt,...);

/**
 * 从文件导入数据。
*/
ssize_t abcdk_buffer_import(abcdk_buffer_t *buf,int fd);

/**
 * 从文件导入数据。
 * 
 * 阻塞模式的句柄，可能会因为导入数据不足而阻塞。
*/
ssize_t abcdk_buffer_import_atmost(abcdk_buffer_t *buf,int fd,size_t howmuch);

/**
 * 导出数据到文件。
*/
ssize_t abcdk_buffer_export(abcdk_buffer_t *buf,int fd);

/**
 * 导出数据到文件。
*/
ssize_t abcdk_buffer_export_atmost(abcdk_buffer_t *buf,int fd,size_t howmuch);

__END_DECLS


#endif //ABCDK_UTIL_BUFFER_H