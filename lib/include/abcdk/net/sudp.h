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
#include "abcdk/openssl/cipher.h"

__BEGIN_DECLS

/**简单的UDP环境。 */
typedef struct _abcdk_sudp abcdk_sudp_t;

/** 
 * 配置。
*/
typedef struct _abcdk_sudp_config
{
    /**共享密钥。*/
    const char *aes_key_file;

    /**监听地址。*/
    abcdk_sockaddr_t listen;

    /**多播地址。*/
    abcdk_mreq_t mreq;

    /**
     * 输出队列丢包最小阈值。 
     * 
     * @note 有效范围：200~600，默认：200
    */
    int out_min_th;

    /**
     * 输出队列丢包最大阈值。 
     * 
     * @note 有效范围：400~800，默认：400
    */
    int out_max_th;

    /**
     * 输出队列丢包权重因子。
     * 
     * @note 有效范围：1~99，默认：2 
    */
    int out_weight;

    /**
     * 输出队列丢包概率因子。 
     * 
     * @note 有效范围：1~99，默认：2 
    */
    int out_prob;

    /**
     * 输入数据到达通知回调函数。
     * 
     * @param remote 远程地址。
     */
    void (*input_cb)(void *opaque,abcdk_sockaddr_t *remote, const void *data, size_t size);

    /** 环境指针。*/
    void *opaque;

} abcdk_sudp_config_t;

/** 停止。*/
void abcdk_sudp_stop(abcdk_sudp_t **ctx);

/**
 * 启动。
 * 
 * @param [in] cfg 配置。
 * 
*/
abcdk_sudp_t *abcdk_sudp_start(abcdk_sudp_config_t *cfg);

/**
 * 投递数据。
 * 
 * @note 投递的数据对象将被托管，应用层不可以继续访问数据对象。
 * 
 * @param [in] data 数据和地址。data->pptrs[0] 数据，data->pptrs[1] 地址。注：仅做指针复制，不会改变对象的引用计数。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sudp_post(abcdk_sudp_t *ctx,abcdk_object_t *data);

/**
 * 投递数据。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sudp_post_buffer(abcdk_sudp_t *ctx,abcdk_sockaddr_t *remote, const void *data,size_t size);

__END_DECLS

#endif //ABCDK_NET_SUDP_H