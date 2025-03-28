/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_TORCH_CONTEXT_H
#define ABCDK_TORCH_CONTEXT_H

#include "abcdk/util/trace.h"
#include "abcdk/torch/torch.h"

__BEGIN_DECLS

/**上下文环境。*/
typedef struct _abcdk_torch_context
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

} abcdk_torch_context_t;

/**锁毁。*/
void abcdk_torch_context_destroy_host(abcdk_torch_context_t **ctx);

/**锁毁。*/
void abcdk_torch_context_destroy_cuda(abcdk_torch_context_t **ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_context_destroy abcdk_torch_context_destroy_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_context_destroy abcdk_torch_context_destroy_host
#endif //

/**创建。*/
abcdk_torch_context_t *abcdk_torch_context_create_host(int id, int flag);

/**创建。*/
abcdk_torch_context_t *abcdk_torch_context_create_cuda(int id, int flag);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_context_create abcdk_torch_context_create_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_context_create abcdk_torch_context_create_host
#endif //


/**
 * 设备环境绑定到当前线程。
 * 
 * @note 仅对当前线程有效，其它线程不可见。
 * 
 * @param [in] ctx 环境指针。NULL(0) 解除绑定。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_torch_context_current_set_host(abcdk_torch_context_t *ctx);

/**
 * 设备环境绑定到当前线程。
 * 
 * @note 仅对当前线程有效，其它线程不可见。
 * 
 * @param [in] ctx 环境指针。NULL(0) 解除绑定。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_torch_context_current_set_cuda(abcdk_torch_context_t *ctx);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_context_current_set abcdk_torch_context_current_set_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_context_current_set abcdk_torch_context_current_set_host
#endif //

/**
 * 从当前线程获取设备环境。
 * 
 * @note 仅对当前线程有效，其它线程不可见。
*/
abcdk_torch_context_t *abcdk_torch_context_current_get_host();

/**
 * 从当前线程获取设备环境。
 * 
 * @note 仅对当前线程有效，其它线程不可见。
*/
abcdk_torch_context_t *abcdk_torch_context_current_get_cuda();

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_context_current_get abcdk_torch_context_current_get_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_context_current_get abcdk_torch_context_current_get_host
#endif //



__END_DECLS


#endif //ABCDK_TORCH_CONTEXT_H