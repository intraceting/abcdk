/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_HEXDUMP_H
#define ABCDK_UTIL_HEXDUMP_H

#include "general.h"
#include "allocator.h"
#include "tree.h"

/**
 * 十六进制格式选项。
*/
typedef struct _abcdk_hexdump_option
{
    /** 标志。*/
    uint32_t flag;

/** 显示地址。*/
#define ABCDK_HEXDEMP_SHOW_ADDR     0x0001

/** 显示字符。*/
#define ABCDK_HEXDEMP_SHOW_CHAR     0x0002

    /** 宽度。默认: 16 */
    size_t width;

    /** 关键字，NULL(0)或无调色板时 忽略。*/
    abcdk_allocator_t *keyword;

    /** 
     * 调色板，NULL(0) 忽略。
     * 
     * 当调色板的颜色数量少于关键字时，则循环使用颜色。
    */
    abcdk_allocator_t *palette;

}abcdk_hexdump_option_t;

/**
 * 打印十六进制格式。
 * 
 * @return > 0 成功(打印的总长度)，<=0 失败(空间不足或出错)。
*/
ssize_t abcdk_hexdump(FILE *fd, const void *data, size_t size, size_t offset, const abcdk_hexdump_option_t *opt);

/**
 * 打印十六进制格式。
 * 
 * @return > 0 成功(打印的总长度)，<=0 失败(空间不足或出错)。
*/
ssize_t abcdk_hexdump2(const char *file, const void *data, size_t size,size_t offset, const abcdk_hexdump_option_t *opt);
/*------------------------------------------------------------------------------------------------*/


#endif //
