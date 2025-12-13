/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_HEXDUMP_H
#define ABCDK_UTIL_HEXDUMP_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/tree.h"

__BEGIN_DECLS

/**
 * 十六进制格式选项.
*/
typedef struct _abcdk_hexdump_option
{
    /** 进制*/
    int base;

/** 十六进制.*/
#define ABCDK_HEXDEMP_BASE_HEX      0

/** 十进制.*/
#define ABCDK_HEXDEMP_BASE_DEC      1

/** 八进制.*/
#define ABCDK_HEXDEMP_BASE_OCT      2

    /** 标志.*/
    uint32_t flag;

/** 显示地址.*/
#define ABCDK_HEXDEMP_SHOW_ADDR     0x0001

/** 显示字符.*/
#define ABCDK_HEXDEMP_SHOW_CHAR     0x0002

    /** 宽度.默认: 16 */
    size_t width;

    /** 关键字, NULL(0)或无调色板时 忽略.*/
    abcdk_object_t *keyword;

    /** 
     * 调色板, NULL(0) 忽略.
     * 
     * @note 当调色板的颜色数量少于关键字时, 则循环使用颜色.
    */
    abcdk_object_t *palette;

}abcdk_hexdump_option_t;

/**
 * 打印十六进制格式.
 * 
 * @return > 0 成功(打印的总长度), <=0 失败(空间不足或出错).
*/
ssize_t abcdk_hexdump(FILE *fd, const void *data, size_t size, size_t offset, const abcdk_hexdump_option_t *opt);

/**
 * 打印十六进制格式.
 * 
 * @return > 0 成功(打印的总长度), <=0 失败(空间不足或出错).
*/
ssize_t abcdk_hexdump2(const char *file, const void *data, size_t size,size_t offset, const abcdk_hexdump_option_t *opt);

__END_DECLS

#endif //
