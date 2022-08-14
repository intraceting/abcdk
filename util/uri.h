/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_URI_H
#define ABCDK_UTIL_URI_H

#include "util/general.h"
#include "util/object.h"

/**
 * URI的字段索引。
*/
typedef enum _abcdk_uri_field
{
    /** 协议名称*/
    ABCDK_URI_SCHEME = 0,
#define ABCDK_URI_SCHEME ABCDK_URI_SCHEME

    /** 用户名称*/
    ABCDK_URI_USER = 1,
#define ABCDK_URI_USER ABCDK_URI_USER

    /** 访问密码*/
    ABCDK_URI_PSWD = 2,
#define ABCDK_URI_PSWD ABCDK_URI_PSWD

    /** 主机地址(包括端口)*/
    ABCDK_URI_HOST = 3,
#define ABCDK_URI_HOST ABCDK_URI_HOST

    /** 资源路径*/
    ABCDK_URI_PATH = 4,
#define ABCDK_URI_PATH ABCDK_URI_PATH

} abcdk_uri_field_t;

/** URI拆分。*/
abcdk_object_t *abcdk_uri_split(const char *uri);

#endif //ABCDK_UTIL_URI_H