/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_TAR_H
#define ABCDK_UTIL_TAR_H

#include "abcdk/util/general.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS

/**
 * TAR的块长度(512Bytes).
*/
#define ABCDK_TAR_BLOCK_SIZE 512

/**
 * TAR头部信息.
 * 
 * 512 Bytes.
 */
typedef union _abcdk_tar_hdr
{
    /**/
    char fill[ABCDK_TAR_BLOCK_SIZE];

    /**
     * posix header.
    */
    struct
    {
        char name[100];         /*   0 */
        char mode[8];           /* 100 */
        char uid[8];            /* 108 */
        char gid[8];            /* 116 */
        char size[12];          /* 124 */
        char mtime[12];         /* 136 */
        char chksum[8];         /* 148 */
        char typeflag;          /* 156 */
        char linkname[100];     /* 157 */
        char magic[TMAGLEN];    /* 257 */
        char version[TVERSLEN]; /* 263 */
        char uname[32];         /* 265 */
        char gname[32];         /* 297 */
        char devmajor[8];       /* 329 */
        char devminor[8];       /* 337 */
        char prefix[155];       /* 345 */
                                /* 500 */
    } posix;

    /**
     * ustar header.
    */
    struct
    {
        char name[100];         /*   0 */
        char mode[8];           /* 100 */
        char uid[8];            /* 108 */
        char gid[8];            /* 116 */
        char size[12];          /* 124 */
        char mtime[12];         /* 136 */
        char chksum[8];         /* 148 */
        char typeflag;          /* 156 */
        char linkname[100];     /* 157 */
        char magic[TMAGLEN];    /* 257 */
        char version[TVERSLEN]; /* 263 */
        char uname[32];         /* 265 */
        char gname[32];         /* 297 */
        char devmajor[8];       /* 329 */
        char devminor[8];       /* 337 */
        char prefix[131];       /* 345 */
        char atime[12];         /* 476 */
        char ctime[12];         /* 488 */
                                /* 500 */
    } ustar;

} __attribute__((packed)) abcdk_tar_hdr;

/*
 * gnu tar extensions:
*/

/** 长链接, 长路径标识.*/
#define ABCDK_USTAR_LONGNAME_MAGIC "././@LongLink"
/** 标识长度(字节, 包括结束符号). */
#define ABCDK_USTAR_LONGNAME_MAGIC_LEN 14
/** 长链接类型.*/
#define ABCDK_USTAR_LONGLINK_TYPE 'K'
/** 长路径类型.*/
#define ABCDK_USTAR_LONGNAME_TYPE 'L'

/*
 * USTAR格式如下: 
 *
 * |长链接头部+长链接实体 (可选) |长路径头部+长路径实体 (可选) |头部 |实体(可选) |块对齐 |
*/

/** 
 * TAR格式专用的数值转字符.
 * 
 * @param len 输出长度(包含结束字符).
 * 
 * @return 0 成功, -1 失败(空间不足).
*/
int abcdk_tar_num2char(uintmax_t val, char *buf, size_t len);

/** 
 * TAR格式专用的字符转数值.
 * 
 * @param len 输入长度(包含结束字符).
 * 
 * @return 0 成功, -1 失败(字符中包含非法字符或数值溢出).
*/
int abcdk_tar_char2num(const char *buf, size_t len, uintmax_t *val);

/**
 * 计算TAR头部较验和.
*/
uint32_t abcdk_tar_calc_checksum(abcdk_tar_hdr *hdr);

/** 
 * 提取TAR头部中的较验和字段.
*/
uint32_t abcdk_tar_get_checksum(abcdk_tar_hdr *hdr);

__END_DECLS

#endif // ABCDK_UTIL_TAR_H