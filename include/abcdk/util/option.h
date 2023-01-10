/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_OPTION_H
#define ABCDK_UTIL_OPTION_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS

/** 选项。*/
typedef struct _abcdk_option abcdk_option_t;

/** 释放对象。*/
void abcdk_option_free(abcdk_option_t **opt);

/** 创建对象。*/
abcdk_option_t *abcdk_option_alloc();

/**
 * 配置一个选项。
 * 
 * @note 支持一对多键值组合，相同键值的次序由添加顺序决定。
 * 
 * @param value 值的指针，允许为NULL(0)。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_option_set(abcdk_option_t *opt, const char *key, const char *value);

/**
 * 配置一个选项。
 * 
 * @note 支持一对多键值组合，相同键值的次序由添加顺序决定。
 * 
 * @param merge 是否合并重复的value。0 不合并，!0 合并。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_option_set2(abcdk_option_t *opt, const char *key, const char *value, int merge);

/**
 * 获取一个选项的值。
 * 
 * @param defval 默认值，允许为NULL(0)。
 * 
 * @return !NULL(0) 成功(值的指针)， NULL(0) 失败(键不存在)。
*/
const char* abcdk_option_get(abcdk_option_t *opt, const char *key,size_t index,const char* defval);

/**
 * 获取一个选项的值(整型)。
*/
int abcdk_option_get_int(abcdk_option_t *opt, const char *key,size_t index,int defval);

/**
 * 获取一个选项的值(长整型)。
*/
long abcdk_option_get_long(abcdk_option_t *opt, const char *key,size_t index,long defval);

/**
 * 获取一个选项的值(长长整型)。
*/
long long abcdk_option_get_llong(abcdk_option_t *opt, const char *key,size_t index,long long defval);

/**
 * 获取一个选项的值(浮点型)。
*/
double abcdk_option_get_double(abcdk_option_t *opt, const char *key,size_t index,double defval);

/**
 * 统计选项值的数量。
 * 
 * @param defval 默认值，可以为NULL(0)。
 * 
 * @return >=0 成功(值的数量)，< 0 失败(键不存在)。
*/
ssize_t abcdk_option_count(abcdk_option_t *opt, const char *key);

/**
 * 检测键是否存在。
 * 
 * @return !0 存在， 0 不存在。
*/
#define abcdk_option_exist(opt, key) (abcdk_option_count((opt), (key)) >= 0 ? 1 : 0)

/**
 * 删除一个选项和值。
 * 
 * @return 0 成功，-1 失败(键不存在)。
*/
int abcdk_option_remove(abcdk_option_t *opt, const char *key);

/**
 * 格式化打印。
 * 
 * @param hyphens 连字符，NULL(0) KEY和VALUE分行打印。
 * 
 * @return >=0 成功(输出的长度)，< 0 失败。
*/
ssize_t abcdk_option_fprintf(abcdk_option_t *opt,FILE *fp,const char *hyphens);

/**
 * 格式化打印。
 * 
 * @return >=0 成功(输出的长度)，< 0 失败。
*/
ssize_t abcdk_option_snprintf(abcdk_option_t *opt,char* buf,size_t max,const char *hyphens);

__END_DECLS

#endif //ABCDK_UTIL_OPTION_H