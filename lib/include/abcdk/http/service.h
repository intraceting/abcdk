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
#include "abcdk/asio/asynctcp.h"
#include "abcdk/ssl/openssl.h"
#include "abcdk/http/util.h"

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

    /*上级服务器地址。*/
    const char *up_link;

    /*授权存储路径。*/
    const char *auth_path;

    /**
     * 防火墙回调。
     *
     * @param [in] remote 远程地址。
     *
     * @return 0 允许，!0 阻止。
     */
    int (*firewall_cb)(void *opaque, const char *remote);

    /**
     * 新连接通知回调。
     *
     * @param [in] remote 远程地址。
     * @param [out] userdata 用户环境指针。
     *
     */
    void (*accept_cb)(void *opaque, abcdk_object_t *stream, const char *remote, void **userdata);

    /*请求通知回调。*/
    void (*request_cb)(void *opaque, abcdk_object_t *stream, void *userdata);

    /*输出(空闲)通知回调。*/
    void (*output_cb)(void *opaque, abcdk_object_t *stream, void *userdata);

    /**
     * 关闭通知回调。
     * 
     * @note 在这里可以清理用户环境。 
    */
    void (*close_cb)(void *opaque, abcdk_object_t *stream, void *userdata);

} abcdk_http_service_config_t;

/** 销毁。*/
void abcdk_http_service_destroy(abcdk_http_service_t **ctx);

/** 创建。*/
abcdk_http_service_t *abcdk_http_service_create(const abcdk_http_service_config_t *cfg);

/** 
 * 在请求头查找属性值。
 * 
 * @code
 * Method
 * Scheme
 * Host
 * Script
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


__END_DECLS

#endif // ABCDK_HTTP_SERVICE_H