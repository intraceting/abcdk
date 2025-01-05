/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_URL_H
#define ABCDK_UTIL_URL_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/path.h"

__BEGIN_DECLS

/**
 * URL的字段索引。
*/
typedef enum _abcdk_url_field
{
    /** 协议名称。*/
    ABCDK_URL_SCHEME = 0,
#define ABCDK_URL_SCHEME ABCDK_URL_SCHEME

    /** 授权账号。*/
    ABCDK_URL_USER = 1,
#define ABCDK_URL_USER ABCDK_URL_USER

    /** 访问密码。*/
    ABCDK_URL_PSWD = 2,
#define ABCDK_URL_PSWD ABCDK_URL_PSWD

    /** 主机地址(包括端口)。*/
    ABCDK_URL_HOST = 3,
#define ABCDK_URL_HOST ABCDK_URL_HOST

    /** 资源路径(包括参数)。*/
    ABCDK_URL_PATH = 4,
#define ABCDK_URL_PATH ABCDK_URL_PATH

    /** 授权(账号和密码)。*/
    ABCDK_URL_AUTH = 5,
#define ABCDK_URL_AUTH ABCDK_URL_AUTH

    /**参数。*/
    ABCDK_URL_PARAM = 6,
#define ABCDK_URL_PARAM ABCDK_URL_PARAM

    /** 锚点。*/
    ABCDK_URL_ANCHOR = 7,
#define ABCDK_URL_ANCHOR ABCDK_URL_ANCHOR

    /** 标志。*/
    ABCDK_URL_FLAG = 8,
#define ABCDK_URL_FLAG ABCDK_URL_FLAG

    /** 脚本。*/
    ABCDK_URL_SCRIPT = 9
#define ABCDK_URL_SCRIPT ABCDK_URL_SCRIPT

} abcdk_url_field_t;

/** URL拆分。*/
abcdk_object_t *abcdk_url_split(const char *url);

/** URL创建。*/
abcdk_object_t *abcdk_url_create(int max, const char *fmt, ...);

/** 
 * URL编码。
 * 
 * @param [in] src 明文指针。
 * @param [in] slen 明文长度。
 * @param [out] dst 密文指针。
 * @param [in out] dlen 密文长度。
 * @param [in] component 是否为组件。!0 是，0 否。
 * 
 * @return 未编码的明文长度。
*/
ssize_t abcdk_url_encode(const char *src,size_t slen,char *dst,size_t *dlen, int component);

/** 
 * URL编码。
*/
abcdk_object_t *abcdk_url_encode2(const char *src,size_t slen, int component);

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

/** 
 * URL解码。
*/
abcdk_object_t *abcdk_url_decode2(const char *src,size_t slen,int qm_stop);

/**
 * 去掉URL中冗余的信息。
 * 
 * @note 不会检测目录结构是否存在。
 * 
 * @param [in] decrease 缩减的深度。
*/
char *abcdk_url_abspath(char *buf, size_t decrease);

/**
 * 依据环境信息修补URL。
 * 
 * @note 当目标路径为绝对路径时，直接复制。
 * 
 * @param [in] target 目标路径。
 * @param [in] opaque 环境路径。
*/
abcdk_object_t *abcdk_url_fixpath(const char *target, const char *opaque);

__END_DECLS

#endif //ABCDK_UTIL_URL_H