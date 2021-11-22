/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#ifndef ABCDK_UTIL_JSON_H
#define ABCDK_UTIL_JSON_H

#include "abcdk-util/general.h"

#ifdef HAVE_JSON_C
#include <json-c/json.h>
#endif //HAVE_JSON_C

__BEGIN_DECLS

#ifdef _json_h_

/**/
typedef struct json_object json_object;

/**
 * JSON对象释放。
*/
void abcdk_json_unref(json_object **obj);

/**
 * JSON对象引用。
 * 
 * @return !NULL(0) 成功(JSON对象指针)，NULL(0) 失败。
*/
json_object *abcdk_json_refer(json_object *obj);

/**
 * 解析字符转换成JSON对象。
 * 
 * @return !NULL(0) 成功(JSON对象指针)，NULL(0) 失败。
*/
json_object *abcdk_json_parse(const char *str);

/**
 * 转换JSON对象为字符串。
 * 
 * @return !NULL(0) 成功(字符串指针)，NULL(0) 失败。
*/
const char *abcdk_json_string(json_object *obj);


#endif //_json_h_

__END_DECLS

#endif //ABCDK_UTIL_JSON_H

