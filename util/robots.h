/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_ROBOTS_H
#define ABCDK_UTIL_ROBOTS_H

#include "util/mmap.h"
#include "util/tree.h"
#include "util/buffer.h"

__BEGIN_DECLS

/**
 * robots的字段索引。
*/
enum _abcdk_robots_field
{
    /** KEY */
   ABCDK_ROBOTS_KEY = 0,
#define ABCDK_ROBOTS_KEY    ABCDK_ROBOTS_KEY

    /** VALUE */
   ABCDK_ROBOTS_VALUE = 1
#define ABCDK_ROBOTS_VALUE  ABCDK_ROBOTS_VALUE

};

/**
 * 解析robots文本。
 * 
 * @param text 文本指针。
 * @param agent 代理指针，NULL(0) 全局。
 * 
*/
abcdk_tree_t *abcdk_robots_parse_text(const char *text,const char *agent);

/**
 * 解析robots文本。
 * 
 * @param file 文件名(包含路径)。
*/
abcdk_tree_t *abcdk_robots_parse_file(const char *file,const char *agent);

__END_DECLS


#endif //ABCDK_UTIL_ROBOTS_H