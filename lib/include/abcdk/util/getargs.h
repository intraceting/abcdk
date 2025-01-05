/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_GETARGS_H
#define ABCDK_UTIL_GETARGS_H

#include "abcdk/util/general.h"
#include "abcdk/util/option.h"
#include "abcdk/util/io.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS

/**
 * 导入参数。
 * 
 * @note 未关联键的值，使用前缀做为键。
 * 
 * @param prefix 键的前缀字符串的指针。
*/
void abcdk_getargs(abcdk_option_t *opt, int argc, char *argv[]);

/**
 * 从已经打开的文件导入参数。
 * 
 * @note 注释行将被忽略。
 * 
 * @param delim 分割字符。
 * @param note 注释字符。
 * @param argv0 命令字符串的指针，可以为NULL(0)。
 * @param prefix 键的前缀字符串的指针，NULL(0) 连字符模式匹配。
 * 
 * @code 
 * prefixKEY 
 * VALUE
 * VALUE
 * ...
 * 
 * prefixKEY 
 * VALUE
 * 
 * @endcode 
 *  
*/
void abcdk_getargs_fp(abcdk_option_t *opt, FILE *fp, uint8_t delim, char note,const char *argv0);

/**
 * 从文件导入参数。
 * 
 * @note 注释行将被忽略。
 * 
 * @param file 文件名(或带路径的文件名)的指针。
 * 
*/
void abcdk_getargs_file(abcdk_option_t *opt, const char *file, uint8_t delim, char note, const char *argv0);

/**
 * 从文本导入参数。
 * 
 * @param text 文本的指针。
 * @param len 文本的长度。
*/
void abcdk_getargs_text(abcdk_option_t *opt, const char *text, size_t len, uint8_t delim, char note, const char *argv0);

/**
 * 格式化打印。
 *
 * @return >=0 成功(输出的长度)，< 0 失败。
 */
ssize_t abcdk_getargs_fprintf(abcdk_option_t *opt, FILE *fp, const char *delim, const char *pack);

/**
 * 格式化打印。
 *
 * @return >=0 成功(输出的长度)，< 0 失败。
 */
ssize_t abcdk_getargs_snprintf(abcdk_option_t *opt, char *buf, size_t max, const char *delim, const char *pack);

__END_DECLS

#endif //ABCDK_UTIL_GETARGS_H