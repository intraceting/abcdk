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
 * 文件和目录项的字段。
*/
enum _abcdk_dirent_field
{
    /**
     * 名字(包括路径) 字段索引。
    */
   ABCDK_DIRENT_NAME = 0,
#define ABCDK_DIRENT_NAME    ABCDK_DIRENT_NAME

    /**
     * 状态 字段索引。
    */
   ABCDK_DIRENT_STAT = 1
#define ABCDK_DIRENT_STAT    ABCDK_DIRENT_STAT

};

/**
 * 目录扫描。
 * 
 * 扫描的结果会自动生成一个棵“树”。
 * 
 * @warning 如果目录和文件较多，则需要较多的内存。
 * 
 * @param depth 遍历深度。0 只遍历当前目录，>= 1 遍历多级目录。
 * @param onefs 0 不辨别文件系统是否相同，!0 只在同一个文件系统中遍历。
 * 
*/
void abcdk_dirscan(abcdk_tree_t *father,size_t depth, int onefs);

__END_DECLS

#endif //ABCDKUTIL_DIRENT_H