/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_STRING_H
#define ABCDK_UTIL_STRING_H

#include "abcdk/util/defs.h"
#include "abcdk/util/heap.h"

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
 * @warning 克隆的指针需要用abcdk_heap_free释放。
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
 * @param isctype_cb 字符比较函数。返回值：!0 是，0 否。isctype等函数在ctype.h文件中。
 * @param where 0 右端，1 左端，2 两端。
 * 
*/
char* abcdk_strtrim(char* str,int (*isctype_cb)(int c),int where);

/**
 * 字符串分割。
 * 
 * @param str 待分割字符串的指针。可能会被修改。
 * @param delim 分割字符的串指针。全字匹配，并区分大小写。
 * @param saveptr 临时的指针。不支持访问。
 * 
 * @return !NULL(0) 分割后字符串的指针，NULL(0) 结束。
*/
char *abcdk_strtok(char *str, const char *delim, char **saveptr);

/**
 * 检测字符串中的字符类型。
 * 
 * @param isctype_cb 字符比较函数。返回值：!0 是，0 否。isctype等函数在ctype.h文件中。
 * 
 * @return !0 通过，0 未通过。
*/
int abcdk_strtype(const char* str,int (*isctype_cb)(int c));

/**
 * 字符串查找并替换。
 * 
 * @warning 被替换目标较多时效率不高。
 * 
 * @return  !NULL(0) 成功(指针需要用abcdk_heap_free去释放)， NULL(0) 失败。
*/
char* abcdk_strrep(const char* str,const char *src, const char *dst, int caseAb);

/**
 * 字符串匹配。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_fnmatch(const char *str,const char *wildcard,int caseAb,int ispath);

/**
 * 计算字符串长度。
 * 
 * @param width 字符宽度。1：多字节，2：两字节，4：四字节。
*/
size_t abcdk_cslen(const void *str,int width);

__END_DECLS

#endif //ABCDK_UTIL_STRING_H