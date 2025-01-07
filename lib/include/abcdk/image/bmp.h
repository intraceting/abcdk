/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_IMAGE_BMP_H
#define ABCDK_IMAGE_BMP_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"
#include "abcdk/util/endian.h"

__BEGIN_DECLS

/** 
 * BMP file header.
*/
typedef struct _abcdk_bmp_file_hdr
{
    /**
     * 类型。
     * 
     * 'BM' Windows3.1 or later.
     * 'BA' OS/2 Bitmap array.
     * 'CI' OS/2 Color Icon.
     * 'CP' OS/2 Color Pointer.
     * 'IC' OS/2 Icon.
     * 'PT' OS/2 Pointer.
    */
    uint16_t type;

    /** 文件大小(字节)。*/
    uint32_t size;

    /** 保留。*/
    uint16_t reserved1;

    /** 保留。*/
    uint16_t reserved2;

    /**
     * 偏移量(字节)。
     * 
     * 从文件开始到数据实体的偏移量。
    */
    uint32_t offset;

} __attribute__((packed)) abcdk_bmp_file_hdr;

/**
 * BMP information header.
*/
typedef struct _abcdk_bmp_info_hdr
{
    /** BMP 元数据大小(字节)，包括调色板大小(字节)。*/
    uint32_t size;

    /** 宽(像素)*/
    int32_t width;

    /** 
     * 高(像素)
     * 
     * 正数，图像倒放。
     * 负数，图像正放。
     * 
    */
    int32_t height;

    /** 
     * 颜色平面数。
     * 
     * Must be 1. 
     */
    uint16_t planes;

    /** 
     * 像素位宽。
     * 
     * 1: 双色
     * 4: 16色
     * 8: 256色
     * 24: 彩色
     * 32: 彩色
    */
    uint16_t bitcount;

    /** 
     * 压缩标志。
     * 
     * 0: BI_RGB 不压缩
     * 1: BI_RLE8 8比特编码(8位位图)
     * 2: BI_RLE4 4比特编码(4位位图)
     * 3: BI_BITFIELES 16/32位位图
     * 4: BI_JPEG (仅用于打印机)
     * 5: BI_PNG (仅用于打印机)
    */
    uint32_t compression;

    /** 图像大小(字节)。*/
    uint32_t size_image;

    /** 水平分辨率(像素/米)。*/
    int32_t x_meter;

    /** 垂直分辨率(像素/米)。*/
    int32_t y_meter;

    /** 位图实际使用的颜色表中的颜色数。*/
    uint32_t color_used;

    /** 位图显示过程中重要的颜色数。*/
    uint32_t color_important;

} __attribute__((packed)) abcdk_bmp_info_hdr;

/**
 * BMP color palette
*/
typedef struct _abcdk_bmp_clr_pal
{
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
} __attribute__((packed)) abcdk_bmp_clr_pal;

/**
 * 保存BMP图像。
 * 
 * @param [in] bits 像素位宽，支持24、32位。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_bmp_save_fd(int fd, const uint8_t *data, uint32_t stride, uint32_t width, int32_t height, uint8_t bits);
#define abcdk_bmp_save abcdk_bmp_save_fd

/**
 * 保存BMP图像。
*/
int abcdk_bmp_save_file(const char *file, const uint8_t *data, uint32_t stride, uint32_t width, int32_t height, uint8_t bits);
#define abcdk_bmp_save2 abcdk_bmp_save_file

/**
 * 加载BMP图像。
 * 
 * 支持24、32位。
 * 
 * @param [out] buf 缓存区指针。
 * @param [in] size 缓存区大小(字节)。
 * @param [in] align 宽对齐(字节)。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_bmp_load(int fd, uint8_t *buf, size_t size, uint32_t align,
                   uint32_t *stride, uint32_t *width, int32_t *height, uint8_t *bits);

/**
 * 加载BMP图像。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_bmp_load2(const char *file, uint8_t *buf, size_t size, uint32_t align,
                    uint32_t *stride, uint32_t *width, int32_t *height, uint8_t *bits);

__END_DECLS

#endif //ABCDK_UTIL_BMP_H