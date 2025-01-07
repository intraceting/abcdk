/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SHELL_MTAB_H
#define ABCDK_SHELL_MTAB_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/**
 * 挂载信息。
 */
typedef struct _abcdk_mtab_info
{
    /** 文件系统(分区或设备)。*/
    const char *fs;

    /** 挂载点。*/
    const char *mpoint;

    /** 类型。*/
    const char *type;

    /** 选项。*/
    const char *options;

    /** dump。*/
    const char *dump;

    /** pass。*/
    const char *pass;
} abcdk_mtab_info_t;


/**
 * 枚举挂载信息。
*/
void abcdk_mtab_list(abcdk_tree_t *list);


__END_DECLS

#endif //ABCDK_SHELL_MTAB_H