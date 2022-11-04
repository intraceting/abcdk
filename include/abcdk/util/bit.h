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

__BEGIN_DECLS

/** 
 * 比特位读写器。
 * 
 * @warning 以字节的二进制阅读顺序排列。如：0(7)~7(0) 8(7)~15(0) 16(7)~23(0) 24(7)~31(0) ... 
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
 * 读。
 * 
 * @param [in] bits 长度(bit)。
*/
uint64_t abcdk_bit_read(abcdk_bit_t *ctx, uint8_t bits);

/** 写。*/
void abcdk_bit_write(abcdk_bit_t *ctx, uint8_t bits, uint64_t num);

/** 
 * 读。
 * 
 * @warning 不会改变游标位置。
 * 
 * @param [in] offset 偏移量(bit)。
*/
uint64_t abcdk_bit_pread(abcdk_bit_t *ctx,size_t offset, uint8_t bits);

/** 
 * 写。
 * 
 * @warning 不会改变游标位置。
*/
void abcdk_bit_pwrite(abcdk_bit_t *ctx, size_t offset, uint8_t bits, uint64_t num);

__END_DECLS

#endif //ABCDK_UTIL_BIT_H