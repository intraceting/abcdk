/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_UTIL_NDARRAY_H
#define ABCDK_UTIL_NDARRAY_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

/**简单的多维数组。*/
typedef struct _abcdk_ndarray
{
    /**格式。*/
    int fmt;
    /**块数量。*/
    size_t block;
    /**宽(每块)。*/
    size_t width;
    /**高(每块)。*/
    size_t height;
    /**深(每块)。*/
    size_t depth;
    /**宽占用的字节数量(包括对齐字节)。*/
    size_t stride;
    /**单元格占用的字节数量。*/
    size_t cell;
    /**数据指针。*/
    void *data;
} abcdk_ndarray_t;

/**多维数组常量。*/
typedef enum _abcdk_ndarray_constant
{
    /** NCHW (AAA...BBB...CCC)*/
    ABCDK_NDARRAY_NCHW = 1,
#define ABCDK_NDARRAY_NCHW ABCDK_NDARRAY_NCHW

    /** NHWC (ABC...ABC...ABC)*/
    ABCDK_NDARRAY_NHWC = 2,
#define ABCDK_NDARRAY_NHWC ABCDK_NDARRAY_NHWC

    /** 水平翻转。*/
    ABCDK_NDARRAY_FLIP_H = 1,
#define ABCDK_NDARRAY_FLIP_H ABCDK_NDARRAY_FLIP_H

    /** 垂直翻转。*/
    ABCDK_NDARRAY_FLIP_V = 2,
#define ABCDK_NDARRAY_FLIP_V ABCDK_NDARRAY_FLIP_V
}abcdk_ndarray_constant_t;

/**
 * 计算多维数组占用的空间(字节)。
*/
size_t abcdk_ndarray_size(abcdk_ndarray_t *ndarray);

/**
 * 设置多维数组宽度占用的空间(字节)。
 * 
 * @param align 对齐字节。
*/
void abcdk_ndarray_set_stride(abcdk_ndarray_t *ndarray,size_t align);

/**
 * 计算多维数组单元格的偏移量。
 */
size_t abcdk_ndarray_offset(abcdk_ndarray_t *ndarray, size_t n, size_t x, size_t y, size_t z, int flag);



__END_DECLS

#endif // ABCDK_UTIL_NDARRAY_H