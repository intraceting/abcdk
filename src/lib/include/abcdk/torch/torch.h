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


#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <npp.h>
#ifdef __x86_64__
#include <nvjpeg.h>
#endif //__x86_64__
#endif //HAVE_CUDA

__BEGIN_DECLS

#ifndef __cuda_cuda_h__

/** */
typedef struct CUctx_st *CUcontext;

/** 
 * Filtering methods.
 */
typedef enum 
{
    NPPI_INTER_UNDEFINED         = 0,        /**<  Undefined filtering interpolation mode. */
    NPPI_INTER_NN                = 1,        /**<  Nearest neighbor filtering. */
    NPPI_INTER_LINEAR            = 2,        /**<  Linear interpolation. */
    NPPI_INTER_CUBIC             = 4,        /**<  Cubic interpolation. */
    NPPI_INTER_CUBIC2P_BSPLINE,              /**<  Two-parameter cubic filter (B=1, C=0) */
    NPPI_INTER_CUBIC2P_CATMULLROM,           /**<  Two-parameter cubic filter (B=0, C=1/2) */
    NPPI_INTER_CUBIC2P_B05C03,               /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
    NPPI_INTER_SUPER             = 8,        /**<  Super sampling. */
    NPPI_INTER_LANCZOS           = 16,       /**<  Lanczos filtering. */
    NPPI_INTER_LANCZOS3_ADVANCED = 17,       /**<  Generic Lanczos filtering with order 3. */
    NPPI_SMOOTH_EDGE             = (int)0x8000000 /**<  Smooth edge filtering. */
} NppiInterpolationMode; 

#endif //__cuda_cuda_h__


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
int abcdk_torch_init_cuda(uint32_t flags);

/**
 * 获取运行时库的版本号。
 * 
 * @param [out] minor 次版本。NULL(0) 忽略。
 * 
 * @return >=0 主版本，< 0  失败。
*/
int abcdk_torch_get_runtime_version_cuda(int *minor);

/** 
 * 获取设备名称。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_torch_get_device_name_cuda(char name[256], int device);

__END_DECLS

#endif //ABCDK_TORCH_TORCH_H