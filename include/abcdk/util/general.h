/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_GENERAL_H
#define ABCDK_UTIL_GENERAL_H

#include "abcdk/util/defs.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS


/**
 * 数值对齐。
 * 
 * @param align 对齐量。0,1是等价的。
*/
size_t abcdk_align(size_t size,size_t align);

/**
 * 执行一次。
 * 
 * @param status 状态，一般是静态类型。必须初始化为0。
 * @param routine 执行函数。0 成功，!0 失败。
 * 
 * @return  = 0 成功(第一次)，>= 1 成功(已执行)，-1 失败。
*/
int abcdk_once(volatile int* status,int (*routine)(void *opaque),void *opaque);


/**
 * 二进制转十六进制。
 * 
 * @param dst 十六进制数据的指针。可用空间至少是二进制数据长度的两倍。
 * @param src 二进制数的指针。
 * @param size 二进制数据的长度。
 * @param ABC 0 小写，!0 大写。
 * 
 * @return !NULL(0) 成功(十六进制数据的指针)，NULL(0) 失败。
*/
char *abcdk_bin2hex(char* dst,const void *src,size_t size,int ABC);

/**
 * 十六进制转二进制。
 * 
 * @param dst 二进制数据的指针。可用空间至少是十六进制数据长度的二分之一。
 * @param src 十六进制数的指针。
 * @param size 十六进制数据的长度。
 * 
 * @return !NULL(0) 成功(二进制数据的指针)，NULL(0) 失败。
*/
void *abcdk_hex2bin(void *dst,const char* src,size_t size);

/**
 * 循环移位。
 *  
 * @param size 数据长度(节字)。
 * @param bits 移动位数。
 * @param direction 1 由低向高，2 由高向低。
*/
void *abcdk_cyclic_shift(void *data,size_t size,size_t bits, int direction);

/**
 * 匹配环境变量。
 * 
 * @code
 * name<delim>value
 * name<delim> value
 * @endcode
 * 
 * @param [in] line 行数据。
 * @param [in] name 变量名称。
 * @param [in] delim 分割字符。
 * 
 * @return !NULL(0) 成功(值的指针)，NULL(0) 失败。
*/
const char *abcdk_match_env(const char *line, const char *name, uint8_t delim);

__END_DECLS

#endif //ABCDK_UTIL_GENERAL_H