/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_HTTP_H
#define ABCDK_UTIL_HTTP_H

#include "abcdk/util/general.h"
#include "abcdk/util/string.h"
#include "abcdk/util/option.h"
#include "abcdk/util/md5.h"
#include "abcdk/util/url.h"

__BEGIN_DECLS

/** 
 * 翻译状态码描述。
 * 
 * @return !NULL(0) 描述字符串指针，NULL(0) 状态码未找到。
*/
const char *abcdk_http_status_desc(uint32_t code);

/** 
 * 匹配环境变量，返回变量的值。
 * 
 * @return !NULL(0) 环境变量值的指针，NULL(0) 不匹配。
*/
const char *abcdk_http_match_env(const char *line, const char *name);

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
 * @param [out] path 路径。NULL(0) 忽略。
 * @param [out] params 参数。NULL(0) 忽略。
 * @param [out] anchor 锚点。NULL(0) 忽略。
 */
void abcdk_http_parse_request_header0(const char *req, abcdk_object_t **method, abcdk_object_t **location, abcdk_object_t **version,
                                      abcdk_object_t **path, abcdk_object_t **params);

/**
 * 表单解码。
*/
abcdk_option_t *abcdk_http_parse_form(const char *form);

__END_DECLS

#endif //ABCDK_UTIL_HTTP_H