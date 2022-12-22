/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_URL_H
#define ABCDK_UTIL_URL_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"

__BEGIN_DECLS

/**
 * URL的字段索引。
*/
typedef enum _abcdk_url_field
{
    /** 协议名称。*/
    ABCDK_URL_SCHEME = 0,
#define ABCDK_URL_SCHEME ABCDK_URL_SCHEME

    /** 用户名称。*/
    ABCDK_URL_USER = 1,
#define ABCDK_URL_USER ABCDK_URL_USER

    /** 访问密码。*/
    ABCDK_URL_PSWD = 2,
#define ABCDK_URL_PSWD ABCDK_URL_PSWD

    /** 主机地址(包括端口)。*/
    ABCDK_URL_HOST = 3,
#define ABCDK_URL_HOST ABCDK_URL_HOST

    /** 资源路径(包括参数)。*/
    ABCDK_URL_PATH = 4
#define ABCDK_URL_PATH ABCDK_URL_PATH

} abcdk_url_field_t;

/** URL拆分。*/
abcdk_object_t *abcdk_url_split(const char *url);

/** 
 * URL编码。
 * 
 * @param [in] src 明文指针。
 * @param [in] slen 明文长度。
 * @param [out] dst 密文指针。
 * @param [in out] dlen 密文长度。
 * @param [in] component !0 是组件，0 是URL。
 * 
 * @return 未编码的明文长度。
*/
ssize_t abcdk_url_encode(const char *src,size_t slen,char *dst,size_t *dlen, int component);

/** 
 * URL解码。
 * 
 * @param [in] src 密文指针。
 * @param [in] slen 密文长度。
 * @param [out] dst 明文指针。
 * @param [in out] dlen 明文长度。
 * @param [in] qm_stop 遇问号(?)终止。!0 是，0 否。
 * 
 * @return 未解码的密文长度。
*/
ssize_t abcdk_url_decode(const char *src,size_t slen,char *dst,size_t *dlen,int qm_stop);

__END_DECLS

#endif //ABCDK_UTIL_URL_H