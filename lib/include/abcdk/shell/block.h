/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SHELL_BLOCK_H
#define ABCDK_SHELL_BLOCK_H

#include "abcdk/util/general.h"
#include "abcdk/util/dirent.h"
#include "abcdk/shell/mmc.h"
#include "abcdk/shell/scsi.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"

__BEGIN_DECLS

/**
 * 根据块名字查找设备(的路径)。
 * 
 * @return 0 成功，-1 失败(未找到)。
*/
int abcdk_block_find_device(const char *name,char devpath[PATH_MAX]);


__END_DECLS

#endif //ABCDK_SHELL_BLOCK_H