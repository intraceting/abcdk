/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_CUDA_CUDA_H
#define ABCDK_CUDA_CUDA_H

#include "abcdk/util/general.h"
#include "abcdk/util/trace.h"

#ifdef HAVE_CUDA
#include <cuda.h> //强制编译器从指定路径查找，而不是项目里。
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "npp.h"
#ifdef __x86_64__
#include "nvjpeg.h"
#endif //__x86_64__
#endif //HAVE_CUDA

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

/** 
 * 2D Point
 */
typedef struct 
{
    int x;      /**<  x-coordinate. */
    int y;      /**<  y-coordinate. */
} NppiPoint;

/**
 * 2D Size
 * This struct typically represents the size of a a rectangular region in
 * two space.
 */
typedef struct 
{
    int width;  /**<  Rectangle width. */
    int height; /**<  Rectangle height. */
} NppiSize;

/**
 * 2D Rectangle
 * This struct contains position and size information of a rectangle in 
 * two space.
 * The rectangle's position is usually signified by the coordinate of its
 * upper-left corner.
 */
typedef struct
{
    int x;          /**<  x-coordinate of upper left corner (lowest memory address). */
    int y;          /**<  y-coordinate of upper left corner (lowest memory address). */
    int width;      /**<  Rectangle width. */
    int height;     /**<  Rectangle height. */
} NppiRect;

#endif //__cuda_cuda_h__

#endif //ABCDK_CUDA_CUDA_H