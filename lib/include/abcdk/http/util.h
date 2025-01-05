/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_HTTP_UTIL_H
#define ABCDK_HTTP_UTIL_H

#include "abcdk/util/general.h"
#include "abcdk/util/string.h"
#include "abcdk/util/option.h"
#include "abcdk/util/md5.h"
#include "abcdk/util/url.h"
#include "abcdk/util/basecode.h"

__BEGIN_DECLS

/** 
 * 翻译状态码描述。
 * 
 * @return !NULL(0) 描述字符串指针，NULL(0) 状态码未找到。
*/
const char *abcdk_http_status_desc(uint32_t code);

/** 
 * 翻译内容类型描述。
 * 
 * @return !NULL(0) 描述字符串指针，NULL(0) 状态码未找到。
*/
const char *abcdk_http_content_type_desc(const char *ext);

/**
 * 计算授权摘要信息。
 */
void abcdk_http_auth_digest(abcdk_md5_t *ctx, const char *user, const char *pawd,
                            const char *method, const char *url, const char *realm, const char *nonce);

/**
 * 解析请求头部的第一行。
 * 
 * @warning 旧的信息将会被释放。
 * 
 * @param [out] method 方法。NULL(0) 忽略。
 * @param [out] location 定位。NULL(0) 忽略。
 * @param [out] version 版本。NULL(0) 忽略。
 */
void abcdk_http_parse_request_header0(const char *req, abcdk_object_t **method, abcdk_object_t **location, abcdk_object_t **version);

/**
 * 表单解码。
 * 
 * @param form 表单。
*/
void abcdk_http_parse_form(abcdk_option_t *opt,const char *form);

/**
 * 构造块数据。
 * 
 * @param  [in] data 数据，NULL(0) 应答结束块。
 * @param  [in] size 长度，<= 0 应答结束块。
 * 
*/
abcdk_object_t *abcdk_http_chunked_copyfrom(const void *data, size_t size);

/** 
 * 构造块数据。
 * 
 * @param [in] max 格式化数据最大长度。 
 * 
*/
abcdk_object_t *abcdk_http_chunked_vformat(int max, const char *fmt, va_list ap);

/** 
 * 构造块数据。
 * 
*/
abcdk_object_t *abcdk_http_chunked_format(int max, const char *fmt, ...);

/**
 * 解码授权。
 * 
 * @note 仅支持Basic和Digest。
*/
void abcdk_http_parse_auth(abcdk_option_t **opt,const char *auth);

/**
 * 加载授权密码。
 *
 * @param [in] user 用户名。
 * @param [out] pawd 密码(明文)。
 *
 * @return 0 账号存在，-1 账号不存在，-2 账号存在但密码为空。
 */
typedef int (*abcdk_http_auth_load_pawd_cb)(void *opaque, const char *user, char pawd[160]);

/**
 * 验证授权。
 * 
 * @note 需要http-method属性的支持。
 *
 * @return 0 成功，-1 失败，-22 参数错误。
*/
int abcdk_http_check_auth(abcdk_option_t *opt,abcdk_http_auth_load_pawd_cb load_pawd_cb,void *opaque);


__END_DECLS

#endif //ABCDK_HTTP_UTIL_H