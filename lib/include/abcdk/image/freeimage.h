/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_IMAGE_FREEIMAGE_H
#define ABCDK_IMAGE_FREEIMAGE_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"

#if defined(__SQL_H) || defined(__SQLEXT_H)
#error "FreeImage与Unixodbc的头文件有冲突，不能同时引用。如果在同一个项目中同时引用这个两个依赖包，需要在不直接相关的源码中分别包含。"
#endif //defined(__SQL_H) || defined(__SQLEXT_H)

#ifdef HAVE_FREEIMAGE
#include <FreeImage.h>
#endif //HAVE_FREEIMAGE

__BEGIN_DECLS

#ifdef FREEIMAGE_H


/**
 * 销毁环境。
 * 
 * @note 需要与abcdk_fi_init配对使用。
*/
void abcdk_fi_uninit();

/**
 * 初始化环境。
 *  
 * @note 需要与abcdk_fi_uninit配对使用。
*/
void abcdk_fi_init(int load_local_plugins_only);

/**
 * 保存图像。
 * 
 * @param data RGB或RGBX数据指针。
 * @param stride 宽度对齐(字节)。
 * @param width 宽度(像素)。
 * @param height 高度(像素)。
 * @param bits 像素位宽，支持24、32位。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_fi_save(FREE_IMAGE_FORMAT fifmt, int fiflag, int fd, const uint8_t *data,
                  uint32_t stride, uint32_t width, uint32_t height, uint8_t bits);

/**
 * 保存图像。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_fi_save2(FREE_IMAGE_FORMAT fifmt, int fiflag, const char *file, const uint8_t *data,
                   uint32_t stride, uint32_t width, uint32_t height, uint8_t bits);


/**
 * 加载图像。
 * 
 * @return !NULL(0) 成功(图像指针)，NULL(0) 失败。
*/
FIBITMAP *abcdk_fi_load(FREE_IMAGE_FORMAT fifmt,int fiflag,int fd);

/**
 * 加载图像。
 * 
 * @return !NULL(0) 成功(图像指针)，NULL(0) 失败。
*/
FIBITMAP *abcdk_fi_load2(FREE_IMAGE_FORMAT fifmt,int fiflag,const char *file);

#endif //FREEIMAGE_H

__END_DECLS

#endif //ABCDK_IMAGE_FREEIMAGE_H