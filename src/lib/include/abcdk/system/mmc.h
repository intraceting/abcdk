/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SYSTEM_MMC_H
#define ABCDK_SYSTEM_MMC_H

#include "abcdk/util/general.h"
#include "abcdk/util/dirent.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"

__BEGIN_DECLS

/**
 * MMC设备信息。
*/
typedef struct _abcdk_mmc_info
{
    /** 总线。*/
    char bus[NAME_MAX];

    /** 设备类型(SD|MMC)。*/
    char type[NAME_MAX];

    /** Card Identification Register。*/
    char cid[NAME_MAX];

    /** 名称(型号)。*/
    char name[NAME_MAX];

    /** 设备名称(可不能存在)。*/
    char devname[NAME_MAX];

}abcdk_mmc_info_t;

/**
 * 获取MMC设备信息。
 * 
 * @note 不包括bus字段。
 * 
 * @return 0 成功，-1 失败(可能不是MMC设备)。
*/
int abcdk_mmc_get_info(const char *path,abcdk_mmc_info_t *info);

/**
 * 枚举MMC设备。
*/
void abcdk_mmc_list(abcdk_tree_t *list);

/**
 * 观察MMC设备变化。
*/
void abcdk_mmc_watch(abcdk_tree_t **snapshot, abcdk_tree_t **add, abcdk_tree_t **del);

__END_DECLS

#endif //ABCDK_SYSTEM_MMC_H