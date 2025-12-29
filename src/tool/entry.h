/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_TOOL_ENTRY_H
#define ABCDK_TOOL_ENTRY_H

#include "abcdk.h"

__BEGIN_DECLS


int abcdk_tool_odbc(abcdk_option_t *args);
int abcdk_tool_mtx(abcdk_option_t *args);
int abcdk_tool_mt(abcdk_option_t *args);
int abcdk_tool_mp4juicer(abcdk_option_t *args);
int abcdk_tool_mp4dump(abcdk_option_t *args);
int abcdk_tool_hexdump(abcdk_option_t *args);
int abcdk_tool_json(abcdk_option_t *args);
int abcdk_tool_lsscsi(abcdk_option_t *args);
int abcdk_tool_archive(abcdk_option_t *args);
int abcdk_tool_lsmmc(abcdk_option_t *args);
int abcdk_tool_basecode(abcdk_option_t *args);
int abcdk_tool_mcdump(abcdk_option_t *args);
int abcdk_tool_uart(abcdk_option_t *args);

__END_DECLS

#endif //ABCDK_TOOL_ENTRY_H
