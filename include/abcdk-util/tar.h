/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_TAR_H
#define ABCDK_UTIL_TAR_H

#include "abcdk-util/general.h"
#include "abcdk-util/blockio.h"

__BEGIN_DECLS

/**
 * TAR的块长度(512Bytes)。
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

/** long link magic.*/
#define ABCDK_USTAR_LONGNAME_MAGIC "././@LongLink"
/** including NULL(0) byte. */
#define ABCDK_USTAR_LONGNAME_MAGIC_LEN 14
/** Identifies the NEXT file on the tape  as having a long linkname.*/
#define ABCDK_USTAR_LONGLINK_TYPE 'K'
/** Identifies the NEXT file on the tape  as having a long name.*/
#define ABCDK_USTAR_LONGNAME_TYPE 'L'

/**
 * TAR
*/
typedef struct _abcdk_tar
{
    /**
     * 文件句柄。
    */
    int fd;

    /**
     * 缓存。
     * 
     * NULL(0) 自由块大小，!NULL(0) 定长块大小。
    */
    abcdk_buffer_t *buf;

} abcdk_tar_t;

/** 
 * TAR格式专用的数值转字符。
 * 
 * @param len 输出长度(包含结束字符)。
 * 
 * @return 0 成功，-1 失败(空间不足)。
*/
int abcdk_tar_num2char(uintmax_t val, char *buf, size_t len);

/** 
 * TAR格式专用的字符转数值。
 * 
 * @param len 输入长度(包含结束字符)。
 * 
 * @return 0 成功，-1 失败(字符中包含非法字符或数值溢出)。
*/
int abcdk_tar_char2num(const char *buf, size_t len, uintmax_t *val);

/**
 * 计算TAR头部较验和。
*/
uint32_t abcdk_tar_calc_checksum(abcdk_tar_hdr *hdr);

/** 
 * 提取TAR头部中的较验和字段。
*/
uint32_t abcdk_tar_get_checksum(abcdk_tar_hdr *hdr);

/** 
 * 提取TAR头部中的长度字段。
*/
int64_t abcdk_tar_get_size(abcdk_tar_hdr *hdr);

/** 
 * 提取TAR头部中的时间字段。
*/
time_t abcdk_tar_get_mtime(abcdk_tar_hdr *hdr);

/** 
 * 提取TAR头部中的状态字段。
*/
mode_t abcdk_tar_get_mode(abcdk_tar_hdr *hdr);

/** 
 * 提取TAR头部中的UID字段。
*/
uid_t abcdk_tar_get_uid(abcdk_tar_hdr *hdr);

/** 
 * 提取TAR头部中的GID字段。
*/
gid_t abcdk_tar_get_gid(abcdk_tar_hdr *hdr);

/** 
 * 填充TAR头部的字段。
 * 
 * @param name 文件名(包括路径)。长度包括NULL(0)结束符。
 * @param linkname 链接名。长度包括NULL(0)结束符。
*/
void abcdk_tar_fill(abcdk_tar_hdr *hdr, char typeflag,
                   const char name[100], const char linkname[100],
                   int64_t size, time_t time, mode_t mode);

/**
 * 较验TAR头部的较验和是否一致。
 * 
 * @return !0 一致，0 不一致。
*/
int abcdk_tar_verify(abcdk_tar_hdr *hdr, const char *magic, size_t size);

/**
 * 从TAR文件中读数据。
 * 
 * @return > 0 读取的长度，<= 0 读取失败或已到末尾。
*/
ssize_t abcdk_tar_read(abcdk_tar_t *tar, void *data, size_t size);

/**
 * 从TAR文件中读数据对齐差额长度的数据，并丢弃掉。
 * 
 * @return 0 成功，-1 失败(读取失败或已到末尾)。
*/
int abcdk_tar_read_align(abcdk_tar_t *tar, size_t size);

/**
 * 向TAR文件中写数据。
 * 
 * @return > 0 写入的长度，<= 0 写入失败或空间不足。
*/
ssize_t abcdk_tar_write(abcdk_tar_t *tar, const void *data, size_t size);

/**
 * 向TAR文件中写数据对齐差额长度的数据。
 * 
 * @return 0 成功，-1 失败(写入失败或空间不足)。
*/
int abcdk_tar_write_align(abcdk_tar_t *tar, size_t size);

/**
 * 向TAR文件中以块为单位写补齐数据。
 * 
 * @param stuffing 填充物。
 * 
 * @return > 0 缓存数据全部写完，= 0 缓存无数据或无缓存，< 0 写入失败或空间不足(剩余数据在缓存中)。
*/
int abcdk_tar_write_trailer(abcdk_tar_t *tar, uint8_t stuffing);

/**
 * 从TAR文件中读数据TAR头部。
 * 
 * @param name 文件名的指针。
 * @param attr 属性的指针。
 * @param linkname 链接名的指针。
 * 
 * @return 0 成功，-1 失败(不是TAR格式或较验和错误)。
 * 
*/
int abcdk_tar_read_hdr(abcdk_tar_t *tar, char name[PATH_MAX], struct stat *attr, char linkname[PATH_MAX]);

/**
 * 向TAR文件写入TAR头部。
 * 
 * @param name 文件名的指针(包括路径)。
 * @param attr 属性的指针。
 * @param linkname 链接名的指针，可以为NULL(0)。
 * 
 * @return 0 成功，-1 失败(写入失败或空间不足)。
*/
int abcdk_tar_write_hdr(abcdk_tar_t *tar, const char *name, const struct stat *attr, const char *linkname);

__END_DECLS

#endif //ABCDK_UTIL_TAR_H