/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_TOOL_ENTRY_H
#define ABCDK_TOOL_ENTRY_H

#include "util/general.h"
#include "util/option.h"
#include "util/getargs.h"

__BEGIN_DECLS

int abcdk_tool_serial(abcdk_tree_t *args);
int abcdk_tool_robots(abcdk_tree_t *args);
int abcdk_tool_release(abcdk_tree_t *args);
int abcdk_tool_odbc(abcdk_tree_t *args);
int abcdk_tool_mtx(abcdk_tree_t *args);
int abcdk_tool_mt(abcdk_tree_t *args);
int abcdk_tool_mp4juicer(abcdk_tree_t *args);
int abcdk_tool_mp4dump(abcdk_tree_t *args);
int abcdk_tool_html(abcdk_tree_t *args);
int abcdk_tool_hexdump(abcdk_tree_t *args);
int abcdk_tool_json(abcdk_tree_t *args);

__END_DECLS

#endif //ABCDK_TOOL_ENTRY_H
