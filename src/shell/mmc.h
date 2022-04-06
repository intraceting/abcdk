/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_BLOCK_H
#define ABCDK_SHELL_BLOCK_H

#include "util/general.h"
#include "util/dirent.h"

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
 * 枚举MMC设备。
*/
void abcdk_mmc_list(abcdk_tree_t *list);

/**
 * 观察MMC设备变化。
*/
void abcdk_mmc_watch(abcdk_tree_t **snapshot, abcdk_tree_t **add, abcdk_tree_t **del);

__END_DECLS

#endif //ABCDK_SHELL_BLOCK_H