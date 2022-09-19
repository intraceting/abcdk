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

/** 
 * URI编码。
 * 
 * @param [in] src 明文指针。
 * @param [in] slen 明文长度。
 * @param [out] dst 密文指针。
 * @param [in out] dlen 密文长度。
 * @param [in] component !0 是组件，0 是URI。
 * 
 * @return 未编码的明文长度。
*/
ssize_t abcdk_uri_encode(const char *src,size_t slen,char *dst,size_t *dlen, int component);

/** 
 * URI解码。
 * 
 * @param [in] src 密文指针。
 * @param [in] slen 密文长度。
 * @param [out] dst 明文指针。
 * @param [in out] dlen 明文长度。
 * 
 * @return 未解码的密文长度。
*/
ssize_t abcdk_uri_decode(const char *src,size_t slen,char *dst,size_t *dlen);

#endif //ABCDK_UTIL_URI_H