/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_NET_HTTPS_H
#define ABCDK_NET_HTTPS_H

#include "abcdk/util/trace.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/stream.h"
#include "abcdk/util/string.h"
#include "abcdk/util/random.h"
#include "abcdk/util/mmap.h"
#include "abcdk/http/util.h"
#include "abcdk/net/stcp.h"



__BEGIN_DECLS

/**简单的HTTP服务。*/
typedef struct _abcdk_https abcdk_https_t;

/**HTTP会话。*/
typedef struct _abcdk_https_session abcdk_https_session_t;

/**HTTP流。*/
typedef struct _abcdk_https_stream abcdk_https_stream_t;

/**配置。*/
typedef struct _abcdk_https_config
{
    /**环境指针。*/
    void *opaque;

    /**名称。*/
    const char *name;

    /**领域。*/
    const char *realm;

    /**安全方案*/
    int ssl_scheme;

    /**CA证书。*/
    const char *pki_ca_file;

    /**CA路径。*/
    const char *pki_ca_path;

    /**
     * 检查吊销列表。
     * 
     * 0 不检查吊销列表，1 仅检查叶证书的吊销列表，2 检查整个证书链路的吊销列表。
    */
    int pki_chk_crl;

    /**证书。*/
    X509 *pki_use_cert;

    /**私钥。*/
    EVP_PKEY *pki_use_key;

    /**请求数量包最大长度。*/
    size_t req_max_size;

    /**请求数量包临时缓存目录。*/
    const char *req_tmp_path;

    /**是否启用H2协议。*/
    int enable_h2;

    /**
     * 授权存储路径。
     * 
     * @note 文件名是用户名，密码是文件内容。
    */
    const char *auth_path;

    /**跨域服务器地址。*/
    const char *a_c_a_o;

    /**
     * 会话准备回调函数。
     * 
     * @param [out] session 新会话。
     * @param [in] listen 监听会话;
     */
    void (*session_prepare_cb)(void *opaque,abcdk_https_session_t **session,abcdk_https_session_t *listen);

    /**
     * 会话验证回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*session_accept_cb)(void *opaque,abcdk_https_session_t *session,int *result);

    /**
     * 会话就绪回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*session_ready_cb)(void *opaque,abcdk_https_session_t *session);


    /**
     * 会话关闭回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*session_close_cb)(void *opaque,abcdk_https_session_t *session);

    /**
     * 流析构回调函数。
     * 
     * @note NULL(0) 忽略。
     */
    void (*stream_destructor_cb)(void *opaque,abcdk_https_stream_t *stream);

    /**
     * 流构造回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*stream_construct_cb)(void *opaque,abcdk_https_stream_t *stream);

    /**
     * 流关闭回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*stream_close_cb)(void *opaque,abcdk_https_stream_t *stream);
    
    /**流请求通知回调函数。*/
    void (*stream_request_cb)(void *opaque,abcdk_https_stream_t *stream);

    /**
     * 流输出(空闲)回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*stream_output_cb)(void *opaque,abcdk_https_stream_t *stream);

} abcdk_https_config_t;


/** 释放会话。*/
void abcdk_https_session_unref(abcdk_https_session_t **session);

/** 引用会话。*/
abcdk_https_session_t *abcdk_https_session_refer(abcdk_https_session_t *src);

/** 申请会话。*/
abcdk_https_session_t *abcdk_https_session_alloc(abcdk_https_t *ctx);

/** 获取会话的用户环境指针。*/
void *abcdk_https_session_get_userdata(abcdk_https_session_t *session);

/** 
 * 设置会话的用户环境指针。
 * 
 * @return 旧的用户环境指针。
*/
void *abcdk_https_session_set_userdata(abcdk_https_session_t *session,void *userdata);

/** 获取会话的地址。*/
const char *abcdk_https_session_get_address(abcdk_https_session_t *session,int remote);

/** 
 * 设置会话的超时时长。
 * 
 * @param [in] timeout 超时时长(秒)。
*/
void abcdk_https_session_set_timeout(abcdk_https_session_t *session,time_t timeout);

/** 
 * 监听。
 * 
 * @note 在会话关闭前，配置信息必须保持有效且不能更改。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_https_session_listen(abcdk_https_session_t *session,abcdk_sockaddr_t *addr,abcdk_https_config_t *cfg);

/** 销毁。*/
void abcdk_https_destroy(abcdk_https_t **ctx);

/** 创建。*/
abcdk_https_t *abcdk_https_create();

/** 释放流。*/
void abcdk_https_unref(abcdk_https_stream_t **stream);

/** 引用流。*/
abcdk_https_stream_t *abcdk_https_refer(abcdk_https_stream_t *src);

/**获取会话指针。*/
abcdk_https_session_t *abcdk_https_get_session(abcdk_https_stream_t *stream);

/** 获取用户环境指针。*/
void *abcdk_https_get_userdata(abcdk_https_stream_t *stream);

/** 
 * 设置用户环境指针。
 * 
 * @return 旧的用户环境指针。
*/
void *abcdk_https_set_userdata(abcdk_https_stream_t *stream,void *userdata);


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
const char* abcdk_https_request_header_get(abcdk_https_stream_t *stream,const char *key);

/**
 * 在请求头查找属性值。
 * 
 * @return !NULL(0) 成功(属性值的指针)，NULL(0) 失败(不存在)。
*/
const char* abcdk_https_request_header_getline(abcdk_https_stream_t *stream,int line);

/** 
 * 获取请求体和长度。
 * 
 * @param [out] len 长度。
 * 
 * @return !NULL(0) 成功(请求体的指针)，NULL(0) 失败(不存在)。
*/
const char* abcdk_https_request_body_get(abcdk_https_stream_t *stream,size_t *len);

/** 通知应答数据数据已经准备好了。*/
void abcdk_https_response_ready(abcdk_https_stream_t *stream);

/**
 * 设置应答头部。
 * 
 * @note 值最大支持4000字符。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_https_response_header_vset(abcdk_https_stream_t *stream,const char *key, const char *val, va_list ap);

/**
 * 设置应答头部。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_https_response_header_set(abcdk_https_stream_t *stream,const char *key, const char *val, ...);

/**
 * 取消应答头部。
*/
void abcdk_https_response_header_unset(abcdk_https_stream_t *stream,const char *key);

/**
 * 头部应答结束。
 * 
 * @note 直接应答实体也会自动执行。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_https_response_header_end(abcdk_https_stream_t *stream);

/**
 * 应答实体。
 * 
 * @note 数据对象写入成功后将被托管，用户不可以再进行读和写操作。
 * 
 * @param [in] data 数据。NULL(0) 表示应答结束。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_https_response(abcdk_https_stream_t *stream,abcdk_object_t *data);

/**
 * 应答实体。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_https_response_buffer(abcdk_https_stream_t *stream,const void *data, size_t size);

/**
 * 应答实体。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_https_response_format(abcdk_https_stream_t *stream,int max, const char *fmt, ...);

/**
 * 应答实体。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_https_response_vformat(abcdk_https_stream_t *stream,int max, const char *fmt, va_list ap);

/**
 * 授权验证。
 * 
 * @note 仅支持Basic和Digest。
 * 
 * @return 0 通过，< 0 未通过。
*/
int abcdk_https_check_auth(abcdk_https_stream_t *stream);


__END_DECLS

#endif // ABCDK_NET_HTTPS_H