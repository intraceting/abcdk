/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_CUDA_DEVICE_H
#define ABCDK_CUDA_DEVICE_H

#include "abcdk/cuda/cuda.h"

#ifdef __cuda_cuda_h__

__BEGIN_DECLS

/** 
 * 获取用于执行计算的设备编号。
 * 
 * @return >= 0 成功(设备编号)，< 0  失败。
*/
int abcdk_cuda_get_device();

/** 
 * 设置用于执行计算的设备编号。
 * 
 * @param [in] device 设备编号。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_set_device(int device);

/** 
 * 获取设备名称。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_get_device_name(char name[256], int device);

/**
 * 获取运行时库的版本号。
 * 
 * @param [out] minor 次版本。NULL(0) 忽略。
 * 
 * @return >=0 主版本，< 0  失败。
*/
int abcdk_cuda_get_runtime_version(int *minor);

/**销毁。 */
void abcdk_cuda_ctx_destroy(CUcontext *ctx);

/**创建。*/
CUcontext abcdk_cuda_ctx_create(int device, int flag);

__END_DECLS

#endif //__cuda_cuda_h__

#endif //ABCDK_CUDA_DEVICE_H