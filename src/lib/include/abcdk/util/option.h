/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_OPTION_H
#define ABCDK_UTIL_OPTION_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS

/** 选项。*/
typedef struct _abcdk_option abcdk_option_t;

/**
 * 选项迭代器。
*/
typedef struct _abcdk_option_iterator
{
    /**
     * 回显函数。
     * 
     * @return -1 终止，>= 0 继续。
    */
    int (*dump_cb)(const char *key,const char *value, void *opaque);

    /**
     * 环境指针。
    */
    void *opaque;

} abcdk_option_iterator_t;

/** 释放。*/
void abcdk_option_free(abcdk_option_t **opt);

/** 
 * 创建。
 * 
 * @param prefix KEY的前缀，允许为“空”。
*/
abcdk_option_t *abcdk_option_alloc(const char *prefix);

/** 获取KEY的前缀。 */
const char *abcdk_option_prefix(abcdk_option_t *opt);

/**
 * 配置一个选项。
 * 
 * @note 支持一对多键值组合，相同键值的次序由添加顺序决定。
 * @note 如果KEY不包含前缀则自动添加。
 * 
 * @param key 键。不区分大小写。
 * @param value 值，允许为“空”或NULL(0)。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_option_set(abcdk_option_t *opt, const char *key, const char *value);

/**
 * 配置一个选项。
 * 
 * @note VALUE最大支持8000字节。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_option_fset(abcdk_option_t *opt, const char *key, const char *valfmt, ...);

/**
 * 获取一个选项的值。
 * 
 * @param [in] defval 默认值，允许为NULL(0)。
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
 * 扫描选项和值。
 * 
 * @note 深度优先遍。
*/
void abcdk_option_scan(abcdk_option_t *opt,abcdk_option_iterator_t *it);

/** 合并。*/
void abcdk_option_merge(abcdk_option_t *dst,abcdk_option_t *src);


__END_DECLS

#endif //ABCDK_UTIL_OPTION_H