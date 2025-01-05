/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_ICONV_H
#define ABCDK_UTIL_ICONV_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

/**
 * 编码转换
 * 
 * @param remain 未被转换数据长度的指针，NULL(0) 忽略。
 * 
 * @return >= 0 成功(目标数据长度(字节))，-1 失败。
*/
ssize_t abcdk_iconv(iconv_t cd, const char *src, size_t slen, char *dst, size_t dlen,size_t *remain);

/**
 * 编码转换
 * 
 * @return >= 0 成功(目标数据长度(字节))，-1 失败。
*/
ssize_t abcdk_iconv2(const char *from,const char *to, const char *src, size_t slen, char *dst, size_t dlen,size_t *remain);


__END_DECLS

#endif //ABCDK_UTIL_ICONV_H