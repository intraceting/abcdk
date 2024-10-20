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
#include "abcdk/util/worker.h"

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

    /*监听地址。*/
    abcdk_sockaddr_t listen;

    /*多播地址。*/
    abcdk_mreq_t *mreq;

    /**
     * 输入数据到达通知回调函数。
     */
    void (*input_cb)(void *opaque,abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote, const void *data, size_t size);

    /**
     * 环境指针。
    */
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
*/
void abcdk_sudp_post(abcdk_sudp_t *ctx,abcdk_sockaddr_t *remote, const void *data, size_t size);

__END_DECLS

#endif //ABCDK_NET_SUDP_H