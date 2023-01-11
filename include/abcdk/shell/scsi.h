/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_SCSI_H
#define ABCDK_SHELL_SCSI_H

#include "abcdk/util/general.h"
#include "abcdk/util/dirent.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"
#include "abcdk/util/scsi.h"

__BEGIN_DECLS

/**
 * SCSI设备信息。
*/
typedef struct _abcdk_scsi_info
{
    /** 总线。*/
    char bus[NAME_MAX];

    /** 设备类型。*/
    uint32_t type;

    /** 序列号。*/
    char serial[NAME_MAX];

    /** 生产商。*/
    char vendor[NAME_MAX];

    /** 型号。*/
    char model[NAME_MAX];

    /** 修订。*/
    char revision[NAME_MAX];

    /** 设备名称(可不能存在)。*/
    char devname[NAME_MAX];

    /** 设备名称(可不能存在)。*/
    char generic[NAME_MAX];

}abcdk_scsi_info_t;

/**
 * 获取SCSI设备信息。
 * 
 * @note 不包括bus字段。
 * 
 * @return 0 成功，-1 失败(可能不是SCSI设备)。
*/
int abcdk_scsi_get_info(const char *path,abcdk_scsi_info_t *info);

/**
 * 枚举SCSI设备。
*/
void abcdk_scsi_list(abcdk_tree_t *list);

/**
 * 观察SCSI设备变化。
*/
void abcdk_scsi_watch(abcdk_tree_t **snapshot, abcdk_tree_t **add, abcdk_tree_t **del);


__END_DECLS

#endif //ABCDK_SHELL_SCSI_H