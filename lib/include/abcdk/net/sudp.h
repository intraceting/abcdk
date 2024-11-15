/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_NET_SUDP_H
#define ABCDK_NET_SUDP_H

#include "abcdk/util/general.h"
#include "abcdk/util/getargs.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/map.h"
#include "abcdk/util/time.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/spinlock.h"
#include "abcdk/util/rwlock.h"
#include "abcdk/util/worker.h"
#include "abcdk/util/wred.h"
#include "abcdk/util/crc.h"
#include "abcdk/openssl/cipherex.h"

__BEGIN_DECLS

/**简单的UDP环境。 */
typedef struct _abcdk_sudp abcdk_sudp_t;

/**安全方案。*/
typedef enum _abcdk_sudp_ssl_scheme
{
    /**RAW.*/
    ABCDK_SUDP_SSL_SCHEME_RAW = 0,
#define ABCDK_SUDP_SSL_SCHEME_RAW   ABCDK_SUDP_SSL_SCHEME_RAW

    /**SKE(Shared key encryption).*/
    ABCDK_SUDP_SSL_SCHEME_SKE = 1,
#define ABCDK_SUDP_SSL_SCHEME_SKE   ABCDK_SUDP_SSL_SCHEME_SKE

}abcdk_sudp_ssl_scheme_t;

/** 
 * 配置。
*/
typedef struct _abcdk_sudp_config
{
    /**安全方案*/
    int ssl_scheme;

    /**监听地址。*/
    abcdk_sockaddr_t listen_addr;

    /**启用组播。!0 启用，0 禁用。*/
    int mreq_enable;

    /**组播地址。*/
    abcdk_mreqaddr_t mreq_addr;

    /**输入工人数量。*/
    int input_worker;

    /** 环境指针。*/
    void *opaque;

    /**
     * 输入数据到达通知回调函数。
     * 
     * @param remote 远程地址。
     */
    void (*input_cb)(void *opaque,abcdk_sockaddr_t *remote, const void *data, size_t size);


} abcdk_sudp_config_t;

/**销毁*/
void abcdk_sudp_destroy(abcdk_sudp_t **ctx);

/**
 * 创建。
 * 
 * @param [in] cfg 配置。
*/
abcdk_sudp_t *abcdk_sudp_create(abcdk_sudp_config_t *cfg);

/**停止。*/
void abcdk_sudp_stop(abcdk_sudp_t *ctx);

/**
 * 密钥重置。
 * 
 * @param [in] 标志。0x01 接收，0x02 发送。可用“|”运算符同时重置。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sudp_cipher_reset(abcdk_sudp_t *ctx,const uint8_t *key,size_t klen,int flag);

/**
 * 投递数据。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sudp_post(abcdk_sudp_t *ctx,abcdk_sockaddr_t *remote, const void *data,size_t size);

__END_DECLS

#endif //ABCDK_NET_SUDP_H