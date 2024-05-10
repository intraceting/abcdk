/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_RPC_RPC_H
#define ABCDK_RPC_RPC_H

#include "abcdk/util/trace.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/waiter.h"

__BEGIN_DECLS

/**简单的RPC服务。*/
typedef struct _abcdk_rpc abcdk_rpc_t;


/**配置。*/
typedef struct _abcdk_rpc_config
{
    /**环境指针。*/
    void *opaque;

    /**数据请求通知回调函数。*/
    void (*request_cb)(void *opaque, uint64_t mid, const void *data, size_t size);

    /**
     * 数据输出通知回调函数。
     * 
     * @return 0 成功，-1 失败。
    */
    int (*output_cb)(void *opaque, const void *data, size_t size);

} abcdk_rpc_config_t;

/** 销毁。*/
void abcdk_rpc_destroy(abcdk_rpc_t **ctx);

/** 创建。*/
abcdk_rpc_t *abcdk_rpc_create(abcdk_rpc_config_t *cfg);

/**
 * 接收数据。
 * 
 * @note 当 @ref data == NULL 或 @ref size <= 0 时，表示链路中断。
 * 
 * @param [in] data 数据指针。
 * @param [in] size 数据长度。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_rpc_input(abcdk_rpc_t *ctx,const void *data,size_t size);

/**
 * 请求。
 * 
 * @note 当 @ref rsp == NULL 时，表示不需要应答。
 * 
 * @return 0 成功，-1 失败(内存不足)，-2 失败(内存不足或链路中断)，-3 失败(链路中断)。
*/
int abcdk_rpc_request(abcdk_rpc_t *ctx,const void *req,size_t req_size,abcdk_object_t **rsp);

/**
 * 应答。
 * 
 * @return 0 成功，-1 失败(内存不足或链路中断)。
*/
int abcdk_rpc_response(abcdk_rpc_t *ctx, uint64_t mid,const void *data,size_t size);

__END_DECLS


#endif //ABCDK_RPC_RPC_H