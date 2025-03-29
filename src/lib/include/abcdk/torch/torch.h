/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_TORCH_H
#define ABCDK_TORCH_TORCH_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/bmp.h"
#include "abcdk/util/thread.h"
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/opencv.h"


__BEGIN_DECLS

/**主机对象。*/
#define ABCDK_TORCH_TAG_HOST ABCDK_FOURCC_MKTAG('h', 'o', 's', 't')

/**CUDA对象。*/
#define ABCDK_TORCH_TAG_CUDA ABCDK_FOURCC_MKTAG('C', 'U', 'D', 'A')

/**点。*/
typedef struct _abcdk_torch_point
{
    int x;
    int y;
} abcdk_torch_point_t;

/**尺寸。*/
typedef struct _abcdk_torch_size
{
    int width;
    int height;
} abcdk_torch_size_t;

/**区域。*/
typedef struct _abcdk_torch_rect
{
    int x;
    int y;
    int width;
    int height;
} abcdk_torch_rect_t;

/** 
 * 初始化。
 * 
 * @return = 0 成功，< 0  失败。
*/
int abcdk_torch_init_host(uint32_t flags);

/** 
 * 初始化。
 * 
 * @return = 0 成功，< 0  失败。
*/
int abcdk_torch_init_cuda(uint32_t flags);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_init abcdk_torch_init_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_init abcdk_torch_init_host
#endif //

/**
 * 获取运行时库的版本号。
 * 
 * @param [out] minor 次版本。NULL(0) 忽略。
 * 
 * @return >=0 主版本，< 0  失败。
*/
int abcdk_torch_get_runtime_version_host(int *minor);


/**
 * 获取运行时库的版本号。
 * 
 * @param [out] minor 次版本。NULL(0) 忽略。
 * 
 * @return >=0 主版本，< 0  失败。
*/
int abcdk_torch_get_runtime_version_cuda(int *minor);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_get_runtime_version abcdk_torch_get_runtime_version_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_get_runtime_version abcdk_torch_get_runtime_version_host
#endif //

/** 
 * 获取设备名称。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_torch_get_device_name_host(char name[256], int id);

/** 
 * 获取设备名称。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_torch_get_device_name_cuda(char name[256], int id);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_get_device_name abcdk_torch_get_device_name_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_get_device_name abcdk_torch_get_device_name_host
#endif //

__END_DECLS

#endif //ABCDK_TORCH_TORCH_H