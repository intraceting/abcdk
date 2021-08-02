/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDKUTIL_HTML_H
#define ABCDKUTIL_HTML_H

#include "tree.h"
#include "mman.h"

__BEGIN_DECLS

/**
 * HTML的字段索引。
*/
enum _abcdk_html_field
{
    /** Key*/
   ABCDK_HTML_KEY = 0,
#define ABCDK_HTML_KEY  ABCDK_HTML_KEY

    /** Value*/
   ABCDK_HTML_VALUE = 1,
#define ABCDK_HTML_VALUE    ABCDK_HTML_VALUE

};

/**
 * 解析HTML文本。
 * 
 * @param text 文本指针。
 * 
*/
abcdk_tree_t *abcdk_html_parse_text(const char *text);

/**
 * 解析HTML文件。
 * 
 * @param file 文件名(包含路径)。
*/
abcdk_tree_t *abcdk_html_parse_file(const char *file);


__END_DECLS

#endif //ABCDKUTIL_HTML_H