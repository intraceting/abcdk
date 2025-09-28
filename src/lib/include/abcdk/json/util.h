/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#ifndef ABCDK_JSON_UTIL_H
#define ABCDK_JSON_UTIL_H

#include "abcdk/json/json.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS

/** 
 * 格式化。
 * 
 * @param readable 启用更好的效果。!0 过长的字符串会被省略，0 全部打印。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_json_format_from_string(const char *str, size_t depth, int readable, FILE *out);

/** 
 * 格式化。
 * 
 * @param readable 启用更好的效果。!0 过长的字符串会被省略，0 全部打印。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_json_format_from_file(const char *file, size_t depth, int readable, FILE *out);

__END_DECLS

#endif //ABCDK_JSON_UTIL_H

