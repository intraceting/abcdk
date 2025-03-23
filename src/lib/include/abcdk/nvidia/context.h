/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_NVIDIA_CONTEXT_H
#define ABCDK_NVIDIA_CONTEXT_H

#include "abcdk/util/trace.h"
#include "abcdk/util/context.h"
#include "abcdk/nvidia/nvidia.h"

__BEGIN_DECLS


/**创建。*/
abcdk_context_t *abcdk_cuda_context_create(int device, int flag);

/**
 * 绑定到线程。
 *
 * @note 仅对当前线程有效，其它线程不可见。
 *  
 * @return 0 成功，< 0  失败。 
*/
int abcdk_cuda_context_setspecific(abcdk_context_t *ctx);

/**
 * 从线程获取。
 *
 * @note 仅对当前线程有效，其它线程不可见。
*/
abcdk_context_t *abcdk_cuda_context_getspecific();

/**
 * 入栈。
 *
 * @return 0 成功，< 0  失败。 
*/
int abcdk_cuda_context_push(abcdk_context_t *ctx);

/**
 * 出栈。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_context_pop(abcdk_context_t *ctx);

__END_DECLS


#endif //ABCDK_NVIDIA_CONTEXT_H