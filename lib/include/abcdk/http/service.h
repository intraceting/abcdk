/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_HTTP_SERVICE_H
#define ABCDK_HTTP_SERVICE_H

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
typedef struct _abcdk_http_service abcdk_http_service_t;

/**流*/
typedef struct _abcdk_http_service_stream abcdk_http_service_stream_t;

/**配置。*/
typedef struct _abcdk_http_service_config
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

    /*监听地址。*/
    const char *listen;

    /*SSL监听地址。*/
    const char *listen_ssl;

    /*上行数量包最大长度。*/
    size_t up_max_size;

    /*上行数量包临时缓存目录。*/
    const char *up_tmp_path;

    /*最大连接数量。*/
    size_t max_client;

    /*传输超时(秒)。*/
    time_t stimeout;

    /*是否启用H2协议。*/
    int enable_h2;

    /**
     * 加载授权。
     * 
     * @param [in] user 用户名。
     * @param [out] pawd 密码(明文)。
     * 
     * @return 0 账号存在，-1 账号不存在，-2 账号存在但密码为空。
    */
    int (*auth_load_cb)(void *opaque,const char *user,char pawd[160]);

    /**
     * 防火墙回调。
     *
     * @param [in] remote 远程地址。
     *
     * @return 0 允许，!0 阻止。
     */
    int (*firewall_cb)(void *opaque,const char *remote);

    /**
     * 新连接通知回调。
     *
     * @param [in] remote 远程地址。
     * @param [out] userdata 用户环境指针。
     *
     */
    void (*accept_cb)(void *opaque,abcdk_object_t *stream);

    /*请求通知回调。*/
    void (*request_cb)(void *opaque,abcdk_object_t *stream);

    /*输出(空闲)通知回调。*/
    void (*output_cb)(void *opaque,abcdk_object_t *stream);

    /**
     * 关闭通知回调。
     * 
     * @note 在这里可以清理用户环境。 
    */
    void (*close_cb)(void *opaque,abcdk_object_t *stream);

} abcdk_http_service_config_t;

/** 销毁。*/
void abcdk_http_service_destroy(abcdk_http_service_t **ctx);

/** 创建。*/
abcdk_http_service_t *abcdk_http_service_create(const abcdk_http_service_config_t *cfg);


/** 获取用户环境指针。*/
void *abcdk_http_service_get_userdata(abcdk_object_t *stream);

/** 
 * 设置用户环境指针。
 * 
 * @return 旧的用户环境指针。
*/
void *abcdk_http_service_set_userdata(abcdk_object_t *stream,void *userdata);

/** 获取远程地址。*/
const char *abcdk_http_service_address_remote(abcdk_object_t *stream);

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
const char* abcdk_http_service_request_header_get(abcdk_object_t *stream,const char *key);

/**
 * 在请求头查找属性值。
 * 
 * @return !NULL(0) 成功(属性值的指针)，NULL(0) 失败(不存在)。
*/
const char* abcdk_http_service_request_header_getline(abcdk_object_t *stream,int line);

/** 
 * 获取请求体和长度。
 * 
 * @param [out] len 长度。
 * 
 * @return !NULL(0) 成功(请求体的指针)，NULL(0) 失败(不存在)。
*/
const char* abcdk_http_service_request_body_get(abcdk_object_t *stream,size_t *len);

/**
 * 应答头部。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_http_service_response_vheader(abcdk_object_t *stream,uint32_t status, int max, const char *fmt, va_list ap);

/**
 * 应答头部。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_http_service_response_header(abcdk_object_t *stream,uint32_t status, int max, const char *fmt, ...);

/**
 * 应答实体。
 * 
 * @param [in] data 数据。NULL(0) 表示应答结束。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_http_service_response_body(abcdk_object_t *stream,abcdk_object_t *data);

/**
 * 应答实体。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_http_service_response_body_buffer(abcdk_object_t *stream,const void *data,size_t size);

/**
 * 应答无实体。
 * 
 * @param [in] a_c_a_m 允许的方法。NULL(0) 不限。
 * @param [in] a_c_a_o 跨域服务器地址。NULL(0) 不限。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_http_service_response_nobody(abcdk_object_t *stream, uint32_t status,const char *a_c_a_m, const char *a_c_a_o);

/**
 * 应答带实体。
 * 
 * @param [in] type 类型。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_http_service_response(abcdk_object_t *stream, uint32_t status, abcdk_object_t *data, const char *type, const char *a_c_a_o);

/**
 * 应答带实体。
 * 
 * @param [in] type 类型。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_http_service_response_buffer(abcdk_object_t *stream, uint32_t status, const char *data, size_t size, const char *type, const char *a_c_a_o);

/**
 * 应答带实体。
 * 
 * @param [in] type 类型。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_http_service_response_fd(abcdk_object_t *stream, uint32_t status, int fd, const char *type, const char *a_c_a_o);

/**
 * 授权验证。
 * 
 * @note 仅支持Basic和Digest。
 * 
 * @return 0 通过，< 0 未通过。
*/
int abcdk_http_service_check_auth(abcdk_object_t *stream);


__END_DECLS

#endif // ABCDK_HTTP_SERVICE_H