/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_CHARSET_H
#define ABCDK_UTIL_CHARSET_H

#include "util/general.h"

/**
 * 较验UTF-8编码。
 * 
 * @warning 不识别BOM头。
 * 
 * 1 bytes: 0xxxxxxx ;
 * 2 bytes: 110xxxxx 10xxxxxx ;
 * 3 bytes: 1110xxxx 10xxxxxx 10xxxxxx ;
 * 4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx ;
 * 5 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx ;
 * 6 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx ;
 * 
 * @return 符合的长度(字节)。
*/
ssize_t abcdk_verify_utf8(const void *data,size_t max);

/**
 * 较验GBK(CP936)编码。
 * 
 * @note 包括GB2312。
 * 
 * 1 bytes:	0x00–-x7F                         ASCII
 * 2 bytes:	0x81-0xFE   0x40-0xFE(No 0x7F)    GBK
 * 
 * @return 符合的长度(字节)。
*/
ssize_t abcdk_verify_gbk(const void *data,size_t max);

/**
 * 较验UCS-2编码。
 * 
 * @warning 不识别BOM头。
 * @param be 0 小端对齐，!0 大端对齐。
 * 
 * @return 符合的长度(字节)。
*/
ssize_t abcdk_verify_ucs2(const void *data,size_t max,int be);

#endif //ABCDK_UTIL_CHARSET_H