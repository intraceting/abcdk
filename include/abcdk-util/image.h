/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_IMAGE_H
#define ABCDK_UTIL_IMAGE_H

#include "abcdk-util/general.h"

/**
 * 简单的图像结构。
*/
typedef struct _abcdk_image
{
    /** 像素格式。*/
    int pixfmt;

    /** 数据指针。*/
    uint8_t *datas[4];

    /** 宽步长(字节)。*/
    int strides[4];

    /** 宽(像素)。*/
    int width;

    /** 高(像素)。*/
    int height;

}abcdk_image_t;

#endif //ABCDK_UTIL_IMAGE_H