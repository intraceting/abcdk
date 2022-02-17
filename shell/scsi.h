/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_SCSI_H
#define ABCDK_SHELL_SCSI_H

#include "util/general.h"
#include "util/dirent.h"

__BEGIN_DECLS

/**
 * SCSI设备信息。
*/
typedef struct _abcdk_scsi_info
{
    /** 
     * 总线。
     * 
     * Host:Channel:Target:Lun
    */
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

    /** 设备名称。*/
    char devname[NAME_MAX];

    /** 设备名称(sg)。*/
    char generic[NAME_MAX];

}abcdk_scsi_info_t;

/**
 * 枚举SCSI设备。
*/
void abcdk_scsi_list(abcdk_tree_t *list);

__END_DECLS

#endif //ABCDK_SHELL_SCSI_H