/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
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
#include "abcdk/util/rwlock.h"
#include "abcdk/util/worker.h"
#include "abcdk/util/asioex.h"
#include "abcdk/util/nonce.h"
#include "abcdk/util/bit.h"
#include "abcdk/openssl/cipherex.h"


__BEGIN_DECLS

/**简单的UDP环境。 */
typedef struct _abcdk_sudp abcdk_sudp_t;

/**UDP节点。 */
typedef struct _abcdk_sudp_node abcdk_sudp_node_t;

/**常量。*/
typedef enum _abcdk_sudp_constant
{
    /**RAW.*/
    ABCDK_SUDP_SSL_SCHEME_RAW = 0,
#define ABCDK_SUDP_SSL_SCHEME_RAW   ABCDK_SUDP_SSL_SCHEME_RAW

    /**AES-256-GCM.*/
    ABCDK_SUDP_SSL_SCHEME_AES256GCM = 1,
#define ABCDK_SUDP_SSL_SCHEME_AES256GCM   ABCDK_SUDP_SSL_SCHEME_AES256GCM

    /**AES-256-CBC.*/
    ABCDK_SUDP_SSL_SCHEME_AES256CBC = 2
#define ABCDK_SUDP_SSL_SCHEME_AES256CBC   ABCDK_SUDP_SSL_SCHEME_AES256CBC
}abcdk_sudp_constant_t;

/** 
 * 配置。
*/
typedef struct _abcdk_sudp_config
{
    /**安全方案*/
    int ssl_scheme;

    /**绑定地址。*/
    abcdk_sockaddr_t bind_addr;

    /**
     * 绑定设备。
     * 
     * @note 需要root权限支持，否则忽略。
    */
    const char *bind_ifname;

    /**启用组播。!0 是，0 否。*/
    int mreq_enable;

    /**组播地址。*/
    abcdk_mreqaddr_t mreq_addr;

    /**
     * 关闭通知回调函数。
     * 
     * @note NULL(0) 忽略。
    */
    void (*close_cb)(abcdk_sudp_node_t *node);

    /**
     * 输入数据到达通知回调函数。
     * 
     * @param remote 远程地址。
     */
    void (*input_cb)(abcdk_sudp_node_t *node,abcdk_sockaddr_t *remote, const void *data, size_t size);


} abcdk_sudp_config_t;

/**
 * 释放。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_sudp_unref(abcdk_sudp_node_t **node);

/**
 * 引用。
*/
abcdk_sudp_node_t *abcdk_sudp_refer(abcdk_sudp_node_t *src);

/**
 * 申请。
 *
 * @param [in] userdata 用户数据长度。
 * @param [in] free_cb 用户数据销毁函数。
 *
 * @return !NULL(0) 成功(指针)，NULL(0) 失败。
 */
abcdk_sudp_node_t *abcdk_sudp_alloc(abcdk_sudp_t *ctx, size_t userdata, void (*free_cb)(void *userdata));

/**
 * 获取索引。
 * 
 * @note 进程内唯一。
 * 
*/
uint64_t abcdk_sudp_get_index(abcdk_sudp_node_t *node);

/**
 * 获取用户环境指针。
*/
void *abcdk_sudp_get_userdata(abcdk_sudp_node_t *node);

/**
 * 设置超时。
 * 
 * @param timeout 时长(秒)。= 0 禁用。默认：0。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_sudp_set_timeout(abcdk_sudp_node_t *node, time_t timeout);

/**
 * 密钥重置。
 * 
 * @param [in] 标志。0x01 接收，0x02 发送。可用“|”运算符同时重置。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sudp_cipher_reset(abcdk_sudp_node_t *node,const uint8_t *key,size_t klen,int flag);

/**销毁*/
void abcdk_sudp_destroy(abcdk_sudp_t **ctx);

/**
 * 创建。
 * 
 * @note NONCE时间误差值越大占用的内存空间越多。
 * 
 * @param [in] worker 工人(线程)数量。
 * @param [in] diff NONCE时间(秒)误差。0 禁用。
*/
abcdk_sudp_t *abcdk_sudp_create(int worker, int diff);

/**停止。*/
void abcdk_sudp_stop(abcdk_sudp_t *ctx);


/**
 * 登记。
 * 
 * @note 在对象关闭前，配置信息必须保持有效且不能更改。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sudp_enroll(abcdk_sudp_node_t *node, abcdk_sudp_config_t *cfg);

/**
 * 投递数据。
 * 
 * @param [in] size 长度。1~64000有效。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sudp_post(abcdk_sudp_node_t *node,abcdk_sockaddr_t *remote, const void *data,size_t size);

__END_DECLS

#endif //ABCDK_NET_SUDP_H