/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_STRING_H
#define ABCDK_UTIL_STRING_H

#include "abcdk/util/defs.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/object.h"

__BEGIN_DECLS

/**
 * 检查字符是否为8进制数字字符。
 * 
 * @return !0 是，0 否。
*/
int abcdk_isodigit(int c);

/**
 * 字符串克隆。
 * 
 * @note 克隆的指针需要用abcdk_heap_free释放。
*/
char *abcdk_strdup(const char *str);

/**
 * 字符串查找。
 * 
 * @param caseAb 0 不区分大小写，!0 区分大小写。
 * 
 * @return !NULL(0) 匹配字符串的首地址， NULL(0) 未找到。
*/
const char* abcdk_strstr(const char *str, const char *sub,int caseAb);

/**
 * 字符串查找。
 * 
 * @return !NULL(0) 匹配字符串的尾地址之后，NULL(0) 未找到。
*/
const char* abcdk_strstr_eod(const char *str, const char *sub,int caseAb);

/**
 * 字符串比较。
 * 
 * @param caseAb 0 不区分大小写，!0 区分大小写。
 * 
 * @return 1(s1 > s2), 0(s1 = s2), -1(s1 < s2)
*/
int abcdk_strcmp(const char *s1, const char *s2,int caseAb);

/**
 * 字符串比较。
 * 
 * @param caseAb 0 不区分大小写，!0 区分大小写。
 * 
 * @return 1(s1 > s2), 0(s1 = s2), -1(s1 < s2)
*/
int abcdk_strncmp(const char *s1, const char *s2,size_t len,int caseAb);

/** 
 * 字符串修剪。
 * 
 * @note isctype等函数在ctype.h文件中。
 * 
 * @param isctype_cb 字符判定函数，!0 是，0 否。NULL(0) 忽略。
 * @param where 0 右端，1 左端，2 两端。
 * 
*/
char* abcdk_strtrim(char* str,int (*isctype_cb)(int c),int where);

/**
 * 字符串修剪。
 * 
 * @param other 其它自定义字符。NULL(0) 忽略。
 * 
*/
char* abcdk_strtrim2(char* str,int (*isctype_cb)(int c), const char *other,int where);

/** 
 * 字符串分割。
 *  
 * @note 仅逻辑分割。
 * 
 * @param [in out] next 字符串。返回前更新。
 * @param [in] delim 分界符。全字匹配，区分大小写。
 * 
 * @return !NULL(0) 成功(字符串首地址)，NULL(0) 失败(已到末尾)。
*/
const char *abcdk_strtok(const char **next, const char *delim);

/**
 * 字符串分割。
 * 
 * @note 仅逻辑分割。

 * @param [in] skip_space 是否跳过空白字符。0 不跳示，!0 跳过。
 * 
 * @return !NULL(0) 成功(字符串首地址)，NULL(0) 失败(已到末尾)。
*/
const char *abcdk_strtok2(const char **next, const char *delim, int skip_space);

/**
 * 字符串分割并复制。
 * 
 * @note 仅逻辑分割。
 * 
 * @see abcdk_strtok2
 * 
 * @return !NULL(0) 成功，NULL(0) 出错或已到末尾。
*/
abcdk_object_t *abcdk_strtok3(const char **next, const char *delim, int skip_space);

/**
 * 字符串分割为一对。
 * 
 * @see abcdk_strtok2
 * 
 * @return !NULL(0) 成功，NULL(0) 出错或已到末尾。
*/
abcdk_object_t *abcdk_strtok2pair(const char *str, const char *delim);

/**
 * 字符串分割为数组。
 * 
 * @note 仅逻辑分割。
 * @note 最多支持256段。
 * 
 * @see abcdk_strtok2
 * 
 * @return !NULL(0) 成功，NULL(0) 出错或已到末尾。
*/
abcdk_object_t *abcdk_strtok2vector(const char *str, const char *delim);

/**
 * 检测字符串中的字符类型。
 * 
 * @param isctype_cb 字符比较函数。返回值：!0 是，0 否。isctype等函数在ctype.h文件中。
 * 
 * @return !0 通过，0 未通过。
*/
int abcdk_strtype(const char* str,int (*isctype_cb)(int c));

/**
 * 计算字符串长度。
 * 
 * @param width 字符宽度。1：多字节，2：两字节，4：四字节。
*/
size_t abcdk_cslen(const void *str,int width);

__END_DECLS

#endif //ABCDK_UTIL_STRING_H