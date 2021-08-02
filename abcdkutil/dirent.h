/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDKUTIL_DIRENT_H
#define ABCDKUTIL_DIRENT_H

#include "general.h"
#include "tree.h"

__BEGIN_DECLS

/**
 * 目录项的字段索引。
*/
enum _abcdk_dirent_field
{
    /** 
     * 名字(包括路径)。
    */
    ABCDK_DIRENT_NAME = 0,
#define ABCDK_DIRENT_NAME ABCDK_DIRENT_NAME

    /**
     * 属性。
    */
    ABCDK_DIRENT_STAT = 1,
#define ABCDK_DIRENT_STAT ABCDK_DIRENT_STAT

    /**
     * 目录的计数器。
     */
    ABCDK_DIRENT_DIRS = 2,
#define ABCDK_DIRENT_DIRS ABCDK_DIRENT_DIRS

    /**
     * 字符设备的计数器。
     */
    ABCDK_DIRENT_CHRS = 3,
#define ABCDK_DIRENT_CHRS ABCDK_DIRENT_CHRS

    /**
     * 块设备的计数器。
     */
    ABCDK_DIRENT_BLKS = 4,
#define ABCDK_DIRENT_BLKS ABCDK_DIRENT_BLKS

    /**
     * 普通文件的计数器。
     */
    ABCDK_DIRENT_REGS = 5,
#define ABCDK_DIRENT_REGS ABCDK_DIRENT_REGS

    /**
     * 管道的计数器。
     */
    ABCDK_DIRENT_FIFOS = 6,
#define ABCDK_DIRENT_FIFOS ABCDK_DIRENT_FIFOS

    /**
     * 软链接的计数器。
     */
    ABCDK_DIRENT_LNKS = 7,
#define ABCDK_DIRENT_LNKS ABCDK_DIRENT_LNKS

    /**
     * SOCKET的计数器。
     */
    ABCDK_DIRENT_SOCKS = 8
#define ABCDK_DIRENT_SOCKS ABCDK_DIRENT_SOCKS
};

/**
 * 计数器结构。 
*/
typedef struct _abcdk_dirent_counter
{
    /** 数量。*/
    size_t nums;

    /** 大小。*/
    size_t sizes;

} abcdk_dirent_counter_t;

/**
 * 目录过滤器。
*/
typedef struct _abcdk_dirent_filter
{
    /**
     * 匹配函数。
     * 
     * @return 0 成功，1 成功(但不扫描子项)，-1 跳过，-2 终止。
    */
    int (*match_cb)(size_t depth,abcdk_tree_t *node,void *opaque);

    /**
     * 环境指针。
    */
    void *opaque;

} abcdk_dirent_filter_t;

/**
 * 目录扫描。
 * 
 * 扫描的结果会自动生成一个棵“树”。
 * 
 * @warning 如果目录和文件较多，则需要较多的内存。
 * 
*/
abcdk_tree_t *abcdk_dirent_scan(const char *path, abcdk_dirent_filter_t *filter);

__END_DECLS

#endif //ABCDKUTIL_DIRENT_H