/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_PACKAGE_H
#define ABCDK_UTIL_PACKAGE_H

#include "abcdk/util/object.h"
#include "abcdk/util/bit.h"
#include "abcdk/util/lz4.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS

/**简单的封包工具。*/
typedef struct _abcdk_package abcdk_package_t;

/**销毁。*/
void abcdk_package_destroy(abcdk_package_t **ctx);

/**
 * 创建。
 * 
 * @param [in] max 最大容量(字节)。<= 2GB.
 * 
*/
abcdk_package_t *abcdk_package_create(size_t max);

/**加载。*/
abcdk_package_t *abcdk_package_load(const uint8_t *data,size_t size);

/**
 * 转储。
 * 
 * @param [in] compress 是否压缩。0 否，!0 是。
*/
abcdk_object_t *abcdk_package_dump(abcdk_package_t *ctx,int compress);

/**
 * 判断游标是否已在末尾。
 * 
 * @return 0 否，!0 是。
*/
int abcdk_package_eof(abcdk_package_t *ctx);

/**
 * 移动游标。
 * 
 * @param [in] offset 偏移量。< 0 向头部移动， > 0 向末尾移动。
 * 
 * @return 游标移动前的位置。
*/
size_t abcdk_package_seek(abcdk_package_t *ctx,ssize_t offset);

/** 
 * 读数值。
 * 
 * @warning 如果原始数值是有符号的，返回值需强制转换为有符号的数值，然后再进行运算或应用。
 * 
 * @param [in] bits 长度(bit)。
*/
uint64_t abcdk_package_read2number(abcdk_package_t *ctx, uint8_t bits);

/**
 * 读缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_package_read2buffer(abcdk_package_t *ctx, uint8_t *buf,size_t size);

/** 
 * 写数值。
 * 
 * @warning 如果原始数值是有符号的，将被强制转换为无符号的数值，然后才会写入到缓存中。
*/
void abcdk_package_write_number(abcdk_package_t *ctx, uint8_t bits, uint64_t num);

/** 
 * 写缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_package_write_buffer(abcdk_package_t *ctx, const uint8_t *buf,size_t size);

/** 
 * 写字符串。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_package_write_string(abcdk_package_t *ctx, const char *buf,size_t size);

/** 
 * 读数值。
 * 
 * @note 不会改变游标位置。
 * @warning 如果原始数值是有符号的，返回值需强制转换为有符号的数值，然后再进行运算或应用。
 * 
 * @param [in] offset 偏移量(bit)。
*/
uint64_t abcdk_package_pread2number(abcdk_package_t *ctx,size_t offset, uint8_t bits);

/**
 * 读缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_package_pread2buffer(abcdk_package_t *ctx, size_t offset, uint8_t *buf,size_t size);

/** 
 * 写数值。
 * 
 * @note 不会改变游标位置。
 * @warning 如果原始数值是有符号的，将被强制转换为无符号的数值，然后才会写入到缓存中。
 * 
 * @param [in] offset 偏移量(bit)。
*/
void abcdk_package_pwrite_number(abcdk_package_t *ctx, size_t offset, uint8_t bits, uint64_t num);

/** 
 * 写缓存。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_package_pwrite_buffer(abcdk_package_t *ctx, size_t offset, const uint8_t *buf,size_t size);


/** 
 * 写字符串。
 * 
 * @note 当游标指向字节开始位时允许使用。
*/
void abcdk_package_pwrite_string(abcdk_package_t *ctx, size_t offset, const char *buf,size_t size);

__END_DECLS

#endif //ABCDK_UTIL_PACKAGE_H