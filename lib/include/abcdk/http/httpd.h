/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_HTTP_HTTPD_H
#define ABCDK_HTTP_HTTPD_H

#include "abcdk/util/trace.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/stream.h"
#include "abcdk/util/string.h"
#include "abcdk/util/random.h"
#include "abcdk/util/mmap.h"
#include "abcdk/http/util.h"
#include "abcdk/asio/asynctcp.h"
#include "abcdk/ssl/openssl.h"


__BEGIN_DECLS

/**简单的HTTP服务。*/
typedef struct _abcdk_httpd abcdk_httpd_t;

/**HTTP会话。*/
typedef struct _abcdk_httpd_session abcdk_httpd_session_t;

/**配置。*/
typedef struct _abcdk_httpd_config
{
    /*环境指针。*/
    void *opaque;

    /*服务器名称。*/
    const char *server_name;

    /*服务器领域。*/
    const char *server_realm;

    /*CA证书。*/
    const char *ca_file;

    /*CA路径。*/
    const char *ca_path;

    /*证书。*/
    const char *cert_file;

    /*私钥。*/
    const char *key_file;

    /*请求数量包最大长度。*/
    size_t req_max_size;

    /*请求数量包临时缓存目录。*/
    const char *req_tmp_path;

    /*是否启用H2协议。*/
    int enable_h2;

    /**
     * 授权存储路径。
     * 
     * @note 文件名是用户名，密码是文件内容。
    */
    const char *auth_path;

    /*跨域服务器地址。*/
    const char *a_c_a_o;

    /**
     * 会话准备通知回调函数。
     * 
     * @param [out] session 新会话。
     * @param [in] listen 监听会话;
     */
    void (*session_prepare_cb)(void *opaque,abcdk_httpd_session_t **session,abcdk_httpd_session_t *listen);

    /**
     * 会话验证通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*session_accept_cb)(void *opaque,abcdk_httpd_session_t *session,int *result);

    /**
     * 会话就绪通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*session_ready_cb)(void *opaque,abcdk_httpd_session_t *session);

    /**
     * 会话关闭通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*session_close_cb)(void *opaque,abcdk_httpd_session_t *session);

    /**
     * 析构通知回调函数。
     * 
     * @note NULL(0) 忽略。
     */
    void (*stream_destructor_cb)(void *opaque,abcdk_object_t *stream);

    /**
     * 构造通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*stream_construct_cb)(void *opaque,abcdk_object_t *stream);

    /*请求通知回调函数。*/
    void (*stream_request_cb)(void *opaque,abcdk_object_t *stream);

    /**
     * 输出(空闲)通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*stream_output_cb)(void *opaque,abcdk_object_t *stream);

} abcdk_httpd_config_t;

/** 释放会话。*/
void abcdk_httpd_session_unref(abcdk_httpd_session_t **session);

/** 引用会话。*/
abcdk_httpd_session_t *abcdk_httpd_session_refer(abcdk_httpd_session_t *src);

/** 申请会话。*/
abcdk_httpd_session_t *abcdk_httpd_session_alloc(abcdk_httpd_t *ctx);

/** 获取会话的用户环境指针。*/
void *abcdk_httpd_session_get_userdata(abcdk_httpd_session_t *session);

/** 
 * 设置会话的用户环境指针。
 * 
 * @return 旧的用户环境指针。
*/
void *abcdk_httpd_session_set_userdata(abcdk_httpd_session_t *session,void *userdata);

/** 获取会话的地址。*/
const char *abcdk_httpd_session_get_address(abcdk_httpd_session_t *session,int remote);

/** 
 * 设置会话的超时时长。
 * 
 * @param [in] timeout 超时时长(秒)。
*/
void abcdk_httpd_session_set_timeout(abcdk_httpd_session_t *session,time_t timeout);

/** 
 * 监听。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_httpd_session_listen(abcdk_httpd_session_t *session,abcdk_sockaddr_t *addr,abcdk_httpd_config_t *cfg);

/** 销毁。*/
void abcdk_httpd_destroy(abcdk_httpd_t **ctx);

/** 创建。*/
abcdk_httpd_t *abcdk_httpd_create(int max,int cpu);

/**获取会话指针。*/
abcdk_httpd_session_t *abcdk_httpd_get_session(abcdk_object_t *stream);

/** 获取用户环境指针。*/
void *abcdk_httpd_get_userdata(abcdk_object_t *stream);

/** 
 * 设置用户环境指针。
 * 
 * @return 旧的用户环境指针。
*/
void *abcdk_httpd_set_userdata(abcdk_object_t *stream,void *userdata);


/** 
 * 在请求头查找属性值。
 * 
 * @code
 * Method
 * Scheme
 * Host
 * Script
 * ...
 * @endcode
 * 
 * @return !NULL(0) 成功(属性值的指针)，NULL(0) 失败(不存在)。
*/
const char* abcdk_httpd_request_header_get(abcdk_object_t *stream,const char *key);

/**
 * 在请求头查找属性值。
 * 
 * @return !NULL(0) 成功(属性值的指针)，NULL(0) 失败(不存在)。
*/
const char* abcdk_httpd_request_header_getline(abcdk_object_t *stream,int line);

/** 
 * 获取请求体和长度。
 * 
 * @param [out] len 长度。
 * 
 * @return !NULL(0) 成功(请求体的指针)，NULL(0) 失败(不存在)。
*/
const char* abcdk_httpd_request_body_get(abcdk_object_t *stream,size_t *len);

/**
 * 设置应答头部。
 * 
 * @note 值最大支持4000字符。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_httpd_response_header_vset(abcdk_object_t *stream,const char *key, const char *val, va_list ap);

/**
 * 设置应答头部。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_httpd_response_header_set(abcdk_object_t *stream,const char *key, const char *val, ...);

/**
 * 取消应答头部。
*/
void abcdk_httpd_response_header_unset(abcdk_object_t *stream,const char *key);

/**
 * 应答实体。
 * 
 * @param [in] data 数据。NULL(0) 表示应答结束。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_httpd_response(abcdk_object_t *stream,abcdk_object_t *data);

/**
 * 应答实体。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_httpd_response_buffer(abcdk_object_t *stream,const void *data, size_t size);

/**
 * 应答实体。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_httpd_response_format(abcdk_object_t *stream,int max, const char *fmt, ...);

/**
 * 应答实体。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_httpd_response_vformat(abcdk_object_t *stream,int max, const char *fmt, va_list ap);

/**
 * 授权验证。
 * 
 * @note 仅支持Basic和Digest。
 * 
 * @return 0 通过，< 0 未通过。
*/
int abcdk_httpd_check_auth(abcdk_object_t *stream);


__END_DECLS

#endif // ABCDK_HTTP_HTTPD_H