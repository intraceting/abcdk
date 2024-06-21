/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_ASIO_SRPC_H
#define ABCDK_ASIO_SRPC_H

#include "abcdk/util/trace.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/waiter.h"
#include "abcdk/util/bit.h"
#include "abcdk/util/random.h"
#include "abcdk/util/timer.h"
#include "abcdk/ssl/openssl.h"
#include "abcdk/ssl/easyssl.h"
#include "abcdk/asio/asio.h"

__BEGIN_DECLS

/**简单的SRPC服务。*/
typedef struct _abcdk_srpc abcdk_srpc_t;

/**SRPC会话。*/
typedef struct _abcdk_srpc_session abcdk_srpc_session_t;

/**配置。*/
typedef struct _abcdk_srpc_config
{
    /*环境指针。*/
    void *opaque;

    /*安全方案*/
    int ssl_scheme;
    
    /*CA证书。*/
    const char *openssl_ca_file;

    /*CA路径。*/
    const char *openssl_ca_path;

    /*证书。*/
    const char *openssl_cert_file;

    /*私钥。*/
    const char *openssl_key_file;

    /*是否验证对端证书。0 否，!0 是。*/
    int openssl_check_cert;

    /*共享密钥。*/
    const char *easyssl_key_file;

    /*盐长度。*/
    int easyssl_salt_size;

    /**
     * 会话准备通知回调函数。
     * 
     * @param [out] session 新会话。
     * @param [in] listen 监听会话;
     */
    void (*prepare_cb)(void *opaque,abcdk_srpc_session_t **session,abcdk_srpc_session_t *listen);

    /**
     * 会话验证通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*accept_cb)(void *opaque,abcdk_srpc_session_t *session,int *result);

    /**
     * 会话就绪通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*ready_cb)(void *opaque,abcdk_srpc_session_t *session);

    /**
     * 会话关闭通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*close_cb)(void *opaque,abcdk_srpc_session_t *session);

    /**会话数据请求通知回调函数。*/
    void (*request_cb)(void *opaque, abcdk_srpc_session_t *session, uint64_t mid, const void *data, size_t size);

    /**
     * 会话输出(空闲)通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*output_cb)(void *opaque, abcdk_srpc_session_t *session);

} abcdk_srpc_config_t;

/** 释放会话。*/
void abcdk_srpc_unref(abcdk_srpc_session_t **session);

/** 引用会话。*/
abcdk_srpc_session_t *abcdk_srpc_refer(abcdk_srpc_session_t *src);

/** 申请会话。*/
abcdk_srpc_session_t *abcdk_srpc_alloc(abcdk_srpc_t *ctx);

/** 获取会话的用户环境指针。*/
void *abcdk_srpc_get_userdata(abcdk_srpc_session_t *session);

/** 
 * 设置会话的用户环境指针。
 * 
 * @return 旧的用户环境指针。
*/
void *abcdk_srpc_set_userdata(abcdk_srpc_session_t *session,void *userdata);

/** 获取会话的地址。*/
const char *abcdk_srpc_get_address(abcdk_srpc_session_t *session,int remote);

/** 
 * 设置会话的超时时长。
 * 
 * @param [in] timeout 超时时长(秒)。
*/
void abcdk_srpc_set_timeout(abcdk_srpc_session_t *session,time_t timeout);

/** 销毁。*/
void abcdk_srpc_destroy(abcdk_srpc_t **ctx);

/** 创建。*/
abcdk_srpc_t *abcdk_srpc_create(int max,int cpu);

/** 
 * 监听。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_srpc_listen(abcdk_srpc_session_t *session,abcdk_sockaddr_t *addr,abcdk_srpc_config_t *cfg);

/** 
 * 连接。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_srpc_connect(abcdk_srpc_session_t *session,abcdk_sockaddr_t *addr,abcdk_srpc_config_t *cfg);

/**
 * 请求。
 * 
 * @note 当 @ref rsp == NULL 时，表示不需要应答。
 * 
 * @return 0 成功，-1 失败(内存不足)，-2 失败(内存不足或链路中断)，-3 失败(链路中断)。
*/
int abcdk_srpc_request(abcdk_srpc_session_t *session,const void *req,size_t req_size,abcdk_object_t **rsp);

/**
 * 应答。
 * 
 * @return 0 成功，-1 失败(内存不足或链路中断)。
*/
int abcdk_srpc_response(abcdk_srpc_session_t *session, uint64_t mid,const void *data,size_t size);

__END_DECLS

#endif //ABCDK_ASIO_SRPC_H