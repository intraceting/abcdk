/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_ASIO_RPC_H
#define ABCDK_ASIO_RPC_H

#include "abcdk/util/trace.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/waiter.h"
#include "abcdk/util/bit.h"
#include "abcdk/util/random.h"
#include "abcdk/util/timer.h"
#include "abcdk/ssl/openssl.h"
#include "abcdk/asio/asynctcp.h"

__BEGIN_DECLS

/**简单的RPC服务。*/
typedef struct _abcdk_rpc abcdk_rpc_t;

/**RPC会话。*/
typedef struct _abcdk_rpc_session abcdk_rpc_session_t;

/**配置。*/
typedef struct _abcdk_rpc_config
{
    /*环境指针。*/
    void *opaque;

    /*CA证书。*/
    const char *ca_file;

    /*CA路径。*/
    const char *ca_path;

    /*证书。*/
    const char *cert_file;

    /*私钥。*/
    const char *key_file;

    /**
     * 会话准备通知回调函数。
     * 
     * @param [out] session 新会话。
     * @param [in] listen 监听会话;
     */
    void (*prepare_cb)(void *opaque,abcdk_rpc_session_t **session,abcdk_rpc_session_t *listen);

    /**
     * 会话验证通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*accept_cb)(void *opaque,abcdk_rpc_session_t *session,int *result);

    /**
     * 会话就绪通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*ready_cb)(void *opaque,abcdk_rpc_session_t *session);

    /**
     * 会话关闭通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*close_cb)(void *opaque,abcdk_rpc_session_t *session);

    /**会话数据请求通知回调函数。*/
    void (*request_cb)(void *opaque, abcdk_rpc_session_t *session, uint64_t mid, const void *data, size_t size);

    /**
     * 会话输出(空闲)通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*output_cb)(void *opaque, abcdk_rpc_session_t *session);

} abcdk_rpc_config_t;

/** 释放会话。*/
void abcdk_rpc_unref(abcdk_rpc_session_t **session);

/** 引用会话。*/
abcdk_rpc_session_t *abcdk_rpc_refer(abcdk_rpc_session_t *src);

/** 申请会话。*/
abcdk_rpc_session_t *abcdk_rpc_alloc(abcdk_rpc_t *ctx);

/** 获取会话的用户环境指针。*/
void *abcdk_rpc_get_userdata(abcdk_rpc_session_t *session);

/** 
 * 设置会话的用户环境指针。
 * 
 * @return 旧的用户环境指针。
*/
void *abcdk_rpc_set_userdata(abcdk_rpc_session_t *session,void *userdata);

/** 获取会话的地址。*/
const char *abcdk_rpc_get_address(abcdk_rpc_session_t *session,int remote);

/** 
 * 设置会话的超时时长。
 * 
 * @param [in] timeout 超时时长(秒)。
*/
void abcdk_rpc_set_timeout(abcdk_rpc_session_t *session,time_t timeout);

/** 销毁。*/
void abcdk_rpc_destroy(abcdk_rpc_t **ctx);

/** 创建。*/
abcdk_rpc_t *abcdk_rpc_create(int max,int cpu);

/** 
 * 监听。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_rpc_listen(abcdk_rpc_session_t *session,abcdk_sockaddr_t *addr,abcdk_rpc_config_t *cfg);

/** 
 * 连接。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_rpc_connect(abcdk_rpc_session_t *session,abcdk_sockaddr_t *addr,abcdk_rpc_config_t *cfg);

/**
 * 请求。
 * 
 * @note 当 @ref rsp == NULL 时，表示不需要应答。
 * 
 * @return 0 成功，-1 失败(内存不足)，-2 失败(内存不足或链路中断)，-3 失败(链路中断)。
*/
int abcdk_rpc_request(abcdk_rpc_session_t *session,const void *req,size_t req_size,abcdk_object_t **rsp);

/**
 * 应答。
 * 
 * @return 0 成功，-1 失败(内存不足或链路中断)。
*/
int abcdk_rpc_response(abcdk_rpc_session_t *session, uint64_t mid,const void *data,size_t size);

__END_DECLS

#endif //ABCDK_ASIO_RPC_H