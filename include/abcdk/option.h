/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_OPTION_H
#define ABCDK_OPTION_H

#include "abcdk/general.h"
#include "abcdk/tree.h"

__BEGIN_DECLS

/**
 * 选项的字段索引。
*/
enum _abcdk_option_field
{
    /** Key。*/
   ABCDK_OPTION_KEY = 0,
#define ABCDK_OPTION_KEY     ABCDK_OPTION_KEY

    /** Value。*/
   ABCDK_OPTION_VALUE = 0
#define ABCDK_OPTION_VALUE   ABCDK_OPTION_VALUE

};

/**
 * 配置一个选项。
 * 
 * @param value 值的指针，可以为NULL0)。
 * 
 * 支持一对多键值组合，相同键的值次序由添加顺序决定。
*/
int abcdk_option_set(abcdk_tree_t *opt, const char *key, const char *value);

/**
 * 获取一个选项的值。
 * 
 * @param defval 默认值，可以为NULL(0)。
 * 
 * @return !NULL(0) 成功(值的指针)， NULL(0) 失败(键不存在)。
*/
const char* abcdk_option_get(abcdk_tree_t *opt, const char *key,size_t index,const char* defval);

/**
 * 获取一个选项的值(整型)。
*/
int abcdk_option_get_int(abcdk_tree_t *opt, const char *key,size_t index,int defval);

/**
 * 获取一个选项的值(长整型)。
*/
long abcdk_option_get_long(abcdk_tree_t *opt, const char *key,size_t index,long defval);

/**
 * 获取一个选项的值(浮点型)。
*/
double abcdk_option_get_double(abcdk_tree_t *opt, const char *key,size_t index,double defval);

/**
 * 统计选项值的数量。
 * 
 * @param defval 默认值，可以为NULL(0)。
 * 
 * @return >=0 成功(值的数量)，< 0 失败(键不存在)。
*/
ssize_t abcdk_option_count(abcdk_tree_t *opt, const char *key);

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
int abcdk_option_remove(abcdk_tree_t *opt, const char *key);

/**
 * 格式化打印。
 * 
 * @param hyphens 连字符，NULL(0) KEY和VALUE分行打印。
 * 
 * @return >=0 输出的长度，< 0 失败。
*/
ssize_t abcdk_option_fprintf(FILE *fp,abcdk_tree_t *opt,const char *hyphens);

/**
 * 格式化打印。
 * 
 * @return >=0 输出的长度，< 0 失败。
*/
ssize_t abcdk_option_snprintf(char* buf,size_t max,abcdk_tree_t *opt,const char *hyphens);

__END_DECLS

#endif //ABCDK_OPTION_H