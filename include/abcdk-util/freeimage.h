/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_FREEIMAGE_H
#define ABCDK_UTIL_FREEIMAGE_H

#include "abcdk-util/general.h"

#if defined(__SQL_H) || defined(__SQLEXT_H)
#error "There is a conflict in the header files of FreeImage and unixODBC, so we can't use 'include' reference directly."
#endif //defined(__SQL_H) || defined(__SQLEXT_H)

#ifdef HAVE_FREEIMAGE
#include <FreeImage.h>
#endif //HAVE_FREEIMAGE

__BEGIN_DECLS

#ifdef FREEIMAGE_H

/** 检查图像格式是否支持。*/
#define ABCDK_FI_IMGFMT_CHECK(fifmt)   ((fifmt) >= FIF_BMP && (fifmt) <= FIF_JXR)

/**
 * 日志重定向到syslog。
*/
void abcdk_fi_log2syslog();

/**
 * 销毁环境。
 * 
 * @warning 需要与abcdk_fi_init配对使用。
*/
void abcdk_fi_uninit();

/**
 * 初始化环境。
 *  
 * @warning 需要与abcdk_fi_uninit配对使用。
*/
void abcdk_fi_init(int load_local_plugins_only);

/**
 * 保存图像。
 * 
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

#endif //ABCDK_UTIL_FREEIMAGE_H