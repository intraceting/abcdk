/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SYSTEM_DMI_H
#define ABCDK_SYSTEM_DMI_H

#include "abcdk/system/block.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/path.h"
#include "abcdk/util/dirent.h"
#include "abcdk/util/md5.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/fnmatch.h"

__BEGIN_DECLS


/**
 * 计算DMI的哈希值。
 * 
 * @param [in] stuff 填充物。NULL(0) 忽略。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_dmi_hash(uint8_t uuid[16], const char *stuff);

__END_DECLS

#endif //ABCDK_SYSTEM_DMI_H