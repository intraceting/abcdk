/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_NVIDIA_CONTEXT_H
#define ABCDK_NVIDIA_CONTEXT_H

#include "abcdk/util/trace.h"
#include "abcdk/nvidia/nvidia.h"

__BEGIN_DECLS


/**锁毁。*/
void abcdk_cuda_ctx_destroy(CUcontext *ctx);

/**创建。*/
CUcontext abcdk_cuda_ctx_create(int device, int flag);

/**
 * 设备环境入栈到当前线程头部。
 * 
 * @note 仅对当前线程有效，其它线程不可见。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_cuda_ctx_push(CUcontext ctx);

/**
 * 设备环境从当前线程头部出栈。
 * 
 * @note 仅对当前线程有效，其它线程不可见。
*/
CUcontext abcdk_cuda_ctx_pop();

/**
 * 设备环境绑定到当前线程。
 * 
 * @note 仅对当前线程有效，其它线程不可见。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_cuda_ctx_setspecific(CUcontext ctx);

/**
 * 从当前线程获取设备环境。
 * 
 * @note 仅对当前线程有效，其它线程不可见。
*/
CUcontext abcdk_cuda_ctx_getspecific();

__END_DECLS


#endif //ABCDK_NVIDIA_CONTEXT_H