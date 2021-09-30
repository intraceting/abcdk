/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_DIRENT_H
#define ABCDK_UTIL_DIRENT_H

#include "abcdk-util/general.h"
#include "abcdk-util/tree.h"

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
    __off64_t nums;

    /** 大小。*/
    __off64_t sizes;

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

/**
 * 打开目录。
 * 
 * @note 已打开目录进行压栈缓存。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_dirent_open(abcdk_tree_t *dir,const char *path);

/**
 * 读取目录。
 * 
 * @note 如果已经当前目录没有未读取的子项，则关闭当前目录，回退到一个打开的目录。
 * 
 * @return 0 成功，-1 失败(无子项)。
*/
int abcdk_dirent_read(abcdk_tree_t *dir,char file[PATH_MAX]);

__END_DECLS

#endif //ABCDK_UTIL_DIRENT_H