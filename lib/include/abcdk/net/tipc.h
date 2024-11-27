/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_NET_TIPC_H
#define ABCDK_NET_TIPC_H

#include "abcdk/util/trace.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/waiter.h"
#include "abcdk/util/bit.h"
#include "abcdk/util/random.h"
#include "abcdk/util/timer.h"
#include "abcdk/net/stcp.h"

__BEGIN_DECLS

/**简单的TIPC服务。*/
typedef struct _abcdk_tipc abcdk_tipc_t;

/**配置。*/
typedef struct _abcdk_tipc_config
{
    /*环境指针。*/
    void *opaque;

    /*服务ID。*/
    uint64_t id;

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
    
    /**密钥。*/
    const char *pki_key_passwd;

    /*是否验证对端证书。0 否，!0 是。*/
    int pki_check_cert;

    /*共享密钥。*/
    const char *ske_key_file;
    
    /**
     * 节点连接通知回调函数。
     *
     * @note NULL(0) 忽略。
     */
    void (*accept_cb)(void *opaque, const char *address, int *result);

    /**节点下线通知回调函数。*/
    void (*offline_cb)(void *opaque, uint64_t id);

    /**数据请求通知回调函数。*/
    void (*request_cb)(void *opaque, uint64_t id, uint64_t mid, const void *data, size_t size);

    /**订阅数据通知回调函数。*/
    void (*subscribe_cb)(void *opaque, uint64_t id, uint64_t topic, const void *data, size_t size);
} abcdk_tipc_config_t;

/** 销毁。*/
void abcdk_tipc_destroy(abcdk_tipc_t **ctx);

/*
 * 创建。
 * 
 * @note 在环境销毁前，配置信息必须保持有效且不能更改。
 * 
*/
abcdk_tipc_t *abcdk_tipc_create(abcdk_tipc_config_t *cfg);

/**
 * 监听。
 *
 * @return 0 成功，!0 失败。
 */
int abcdk_tipc_listen(abcdk_tipc_t *ctx, abcdk_sockaddr_t *addr);

/**
 * 连接。
 *
 * @return 0 成功，!0 失败。
 */
int abcdk_tipc_connect(abcdk_tipc_t *ctx, const char *location, uint64_t id);

/**
 * 请求。
 * 
 * @param [out] rsp 应答。NULL(0) 忽略。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tipc_request(abcdk_tipc_t *ctx,uint64_t id,const char *data,size_t size,abcdk_object_t **rsp);

/**
 * 应答。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tipc_response(abcdk_tipc_t *ctx,uint64_t id,uint64_t mid, const char *data,size_t size);

/**
 * 订阅。
 * 
 * @param [in] topic 主题。
 * @param [in] unset 取消订阅。0 否，!0 是。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tipc_subscribe(abcdk_tipc_t *ctx,uint64_t topic,int unset);

/**
 * 发布。
 * 
 * @param [in] topic 主题。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_tipc_publish(abcdk_tipc_t *ctx,uint64_t topic, const char *data,size_t size);

__END_DECLS

#endif // ABCDK_NET_TIPC_H