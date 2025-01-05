/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
*/
#ifndef ABCDK_JSON_JSON_H
#define ABCDK_JSON_JSON_H

#include "abcdk/util/general.h"

#ifdef HAVE_JSON_C
#include <json-c/json.h>
#endif //HAVE_JSON_C

__BEGIN_DECLS

#ifdef _json_h_

/**/
typedef struct json_object json_object;
typedef struct array_list array_list;

/**
 * 阅读格式化。
 * 
 * @param better !0 更好的效果(过长的字符串会被省略)，0 全部打印。
 * 
*/
void abcdk_json_readable(FILE *fp,int better,size_t depth,json_object *obj);

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
 * 定位节点。
 * 
 * @note 不会改变子节点的引用计数。
 * 
 * @param ... KEY的指针，NULL(0) 结束。
 * 
 * @return !NULL(0) 成功(JSON对象指针)，NULL(0) 失败。
*/
json_object* abcdk_json_locate(json_object *father,...);

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

/**
 * 添加子节点.
 * 
 * @note 不会改变子节点的引用计数。
*/
void abcdk_json_add(json_object *father,const char *key,json_object *val);

/**
 * 添加新节点(string)。
 * 
 * @note 不会改变子节点的引用计数。
 * @note value的长度最大为1024字节(包括'\0'终止符)。
 * 
 * @param father 父级节点，NULL(0) 仅创建新节点。
 * @param key 键，NULL(0) 仅创建新节点。
 * 
 * @return !NULL(0) 成功(子节点JSON对象指针)，NULL(0) 失败。
*/
json_object *abcdk_json_add_vformat(json_object *father,const char *key,const char *val_fmt,va_list val_args);

/**
 * 添加新节点(string)。
 * 
 * @note 不会改变子节点的引用计数。
 * @note value的长度最大为1024字节(包括'\0'终止符)。
 * 
 * @param father 父级节点，NULL(0) 仅创建新节点。
 * @param key 键，NULL(0) 仅创建新节点。
 * 
 * @return !NULL(0) 成功(子节点JSON对象指针)，NULL(0) 失败。
*/
json_object *abcdk_json_add_format(json_object *father,const char *key,const char *val_fmt,...);

/**
 * 添加新节点(int32)。
 * 
 * @note 不会改变子节点的引用计数。
 * 
 * @param father 父级节点，NULL(0) 仅创建新节点。
 * @param key 键，NULL(0) 仅创建新节点。
 * 
 * @return !NULL(0) 成功(子节点JSON对象指针)，NULL(0) 失败。
*/
json_object *abcdk_json_add_int32(json_object *father,const char *key,int32_t val);

/**
 * 添加新节点(int64)。
 * 
 * @note 不会改变子节点的引用计数。
 * 
 * @param father 父级节点，NULL(0) 仅创建新节点。
 * @param key 键，NULL(0) 仅创建新节点。
 * 
 * @return !NULL(0) 成功(子节点JSON对象指针)，NULL(0) 失败。
*/
json_object *abcdk_json_add_int64(json_object *father,const char *key,int64_t val);

/**
 * 添加新节点(boolean)。
 * 
 * @note 不会改变子节点的引用计数。
 * 
 * @param father 父级节点，NULL(0) 仅创建新节点。
 * @param key 键，NULL(0) 仅创建新节点。
 * 
 * @return !NULL(0) 成功(子节点JSON对象指针)，NULL(0) 失败。
*/
json_object *abcdk_json_add_boolean(json_object *father,const char *key,json_bool val);

/**
 * 添加新节点(double)。
 * 
 * @note 不会改变子节点的引用计数。
 * 
 * @param father 父级节点，NULL(0) 仅创建新节点。
 * @param key 键，NULL(0) 仅创建新节点。
 * 
 * @return !NULL(0) 成功(子节点JSON对象指针)，NULL(0) 失败。
*/
json_object* abcdk_json_add_double(json_object *father,const char *key,double val);


#endif //_json_h_

__END_DECLS

#endif //ABCDK_JSON_JSON_H

