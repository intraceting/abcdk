/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_BIT_H
#define ABCDK_UTIL_BIT_H

#include "abcdk/util/defs.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/object.h"

__BEGIN_DECLS

/** 
 * 比特位读写器。
 * 
 * @note 以字节的二进制阅读顺序排列。如：0(7)~7(0) 8(7)~15(0) 16(7)~23(0) 24(7)~31(0) ... 
*/
typedef struct _abcdk_bit
{
    /**读写游标。*/
    size_t pos;

    /**数据区指针。*/
    void *data;

    /**数据区大小。*/
    size_t size;
    
} abcdk_bit_t;


/**
 * 判断游标是否已在末尾。
 * 
 * @return 0 否，!0 是。
*/
int abcdk_bit_eof(abcdk_bit_t *ctx);

/**
 * 移动游标。
 * 
 * @param [in] offset 偏移量。< 0 向头部移动， > 0 向末尾移动。
 * 
 * @return 游标移动前的位置。
*/
size_t abcdk_bit_seek(abcdk_bit_t *ctx,ssize_t offset);

/** 
 * 读数值。
 * 
 * @warning 如果原始数值是有符号的，返回值需强制转换为有符号的数值，然后再进行运算或应用。
 * 
 * @param [in] bits 长度(bit)。
*/
uint64_t abcdk_bit_read2number(abcdk_bit_t *ctx, uint8_t bits);
#define abcdk_bit_read abcdk_bit_read2number

/**
 * 读缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_bit_read2buffer(abcdk_bit_t *ctx, uint8_t *buf,size_t size);

/**
 * 读缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
abcdk_object_t *abcdk_bit_read2object(abcdk_bit_t *ctx, size_t size);

/** 
 * 写数值。
 * 
 * @warning 如果原始数值是有符号的，将被强制转换为无符号的数值，然后才会写入到缓存中。
 * 
*/
void abcdk_bit_write_number(abcdk_bit_t *ctx, uint8_t bits, uint64_t num);
#define abcdk_bit_write abcdk_bit_write_number

/** 
 * 写缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_bit_write_buffer(abcdk_bit_t *ctx, const uint8_t *buf,size_t size);

/** 
 * 读数值。
 * 
 * @note 不会改变游标位置。
 * @warning 如果原始数值是有符号的，返回值需强制转换为有符号的数值，然后再进行运算或应用。
 * 
 * @param [in] offset 偏移量(bit)。
*/
uint64_t abcdk_bit_pread2number(abcdk_bit_t *ctx,size_t offset, uint8_t bits);
#define abcdk_bit_pread abcdk_bit_pread2number

/**
 * 读缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_bit_pread2buffer(abcdk_bit_t *ctx, size_t offset, uint8_t *buf,size_t size);

/** 
 * 写数值。
 * 
 * @note 不会改变游标位置。
 * @warning 如果原始数值是有符号的，将被强制转换为无符号的数值，然后才会写入到缓存中。
 * 
 * @param [in] offset 偏移量(bit)。
*/
void abcdk_bit_pwrite_number(abcdk_bit_t *ctx, size_t offset, uint8_t bits, uint64_t num);
#define abcdk_bit_pwrite abcdk_bit_pwrite_number

/** 
 * 写缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_bit_pwrite_buffer(abcdk_bit_t *ctx, size_t offset, const uint8_t *buf,size_t size);

__END_DECLS

#endif //ABCDK_UTIL_BIT_H