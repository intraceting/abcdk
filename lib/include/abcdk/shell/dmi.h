/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SHELL_DMI_H
#define ABCDK_SHELL_DMI_H

#include "abcdk/shell/block.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/md5.h"

__BEGIN_DECLS

/**常量。*/
typedef enum _abcdk_dmi_constant
{
    /**SCSI设备。*/
    ABCDK_DMI_HASH_USE_DEVICE_SCSI = 0x00000001,
#define ABCDK_DMI_HASH_USE_DEVICE_SCSI ABCDK_DMI_HASH_USE_DEVICE_SCSI

    /**MMC设备。*/
    ABCDK_DMI_HASH_USE_DEVICE_MMC = 0x00000002,
#define ABCDK_DMI_HASH_USE_DEVICE_MMC ABCDK_DMI_HASH_USE_DEVICE_MMC

    /**MAC设备。*/
    ABCDK_DMI_HASH_USE_DEVICE_MAC = 0x00000004,
#define ABCDK_DMI_HASH_USE_DEVICE_MAC ABCDK_DMI_HASH_USE_DEVICE_MAC

    /**填充物。*/
    ABCDK_DMI_HASH_USE_STUFF = 0x00000008
#define ABCDK_DMI_HASH_USE_STUFF ABCDK_DMI_HASH_USE_STUFF

}abcdk_dmi_constant_t;

/**
 * 计算DMI哈希值。
 * 
 * @param [in] flag 标志。
 * @param [in] stuff 填充物。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
const uint8_t *abcdk_dmi_hash(uint8_t uuid[16], uint32_t flag, const char *stuff);

__END_DECLS

#endif //ABCDK_SHELL_DMI_H