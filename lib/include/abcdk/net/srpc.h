/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_NET_SRPC_H
#define ABCDK_NET_SRPC_H

#include "abcdk/util/trace.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/waiter.h"
#include "abcdk/util/bit.h"
#include "abcdk/util/random.h"
#include "abcdk/util/timer.h"
#include "abcdk/net/stcp.h"

__BEGIN_DECLS

/**简单的RPC服务。*/
typedef struct _abcdk_srpc abcdk_srpc_t;

/**RPC会话。*/
typedef struct _abcdk_srpc_session abcdk_srpc_session_t;

/**配置。*/
typedef struct _abcdk_srpc_config
{
    /*环境指针。*/
    void *opaque;

    /*安全方案*/
    int ssl_scheme;
    
    /*CA证书。*/
    const char *pki_ca_file;

    /*CA路径。*/
    const char *pki_ca_path;

    /*证书。*/
    const char *pki_cert_file;

    /*私钥。*/
    const char *pki_key_file;

    /*是否验证对端证书。0 否，!0 是。*/
    int pki_check_cert;

    /*共享密钥。*/
    const char *ske_key_file;

    /**密钥算法。*/
    int ske_key_cipher;
    
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
abcdk_srpc_session_t *abcdk_srpc_alloc(abcdk_srpc_t *ctx, size_t userdata, void (*free_cb)(void *userdata));

/** 轨迹输出。*/
void abcdk_srpc_trace_output(abcdk_srpc_session_t *node,int type, const char* fmt,...);

/**
 * 获取索引。
 * 
 * @note 进程内唯一。
 * 
 */
uint64_t abcdk_srpc_get_index(abcdk_srpc_session_t *node);

/** 
 * 获用户环境指针。
 * 
 * @return !NULL(0) 成功(有效)，NULL(0) 失败(无效)。
*/
void *abcdk_srpc_get_userdata(abcdk_srpc_session_t *session);


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

/**
 *  创建。
 * 
 * @param [in] worker 工人(线程)数量。
*/
abcdk_srpc_t *abcdk_srpc_create(int worker);

/** 停止。*/
void abcdk_srpc_stop(abcdk_srpc_t *ctx);

/** 
 * 监听。
 * 
 * @note 在会话关闭前，配置信息必须保持有效且不能更改。
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
 * @note 当 rsp == NULL 时，表示不需要应答。
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


/** 通知数据数据已经准备好了。*/
void abcdk_srpc_output_ready(abcdk_srpc_session_t *session);

__END_DECLS

#endif //ABCDK_NET_SRPC_H