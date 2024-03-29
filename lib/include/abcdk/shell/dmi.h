/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_DMI_H
#define ABCDK_SHELL_DMI_H

#include "abcdk/shell/block.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/md5.h"

__BEGIN_DECLS

/**
 * 获取硬件散列值的标志。
*/
typedef enum _abcdk_dmi_machine_hc_flag
{
    /**SCSI设备。*/
    ABCDK_DMI_MACHINE_HC_DEVICE_SCSI = 0x00000001,
#define ABCDK_DMI_MACHINE_HC_DEVICE_SCSI ABCDK_DMI_MACHINE_HC_DEVICE_SCSI

    /**MMC设备。*/
    ABCDK_DMI_MACHINE_HC_DEVICE_MMC = 0x00000002,
#define ABCDK_DMI_MACHINE_HC_DEVICE_MMC ABCDK_DMI_MACHINE_HC_DEVICE_MMC

    /**MAC设备。*/
    ABCDK_DMI_MACHINE_HC_DEVICE_MAC = 0x00000004,
#define ABCDK_DMI_MACHINE_HC_DEVICE_MAC ABCDK_DMI_MACHINE_HC_DEVICE_MAC

    /**填充物。*/
    ABCDK_DMI_MACHINE_HC_STUFF = 0x00000008
#define ABCDK_DMI_MACHINE_HC_STUFF ABCDK_DMI_MACHINE_HC_STUFF

}abcdk_dmi_machine_hc_flag_t;

/**
 * 获取硬件散列值。
 * 
 * @param [in] flag 标志。
 * @param [in] stuff 填充物。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
const uint8_t *abcdk_dmi_get_machine_hashcode(uint8_t uuid[16], uint32_t flag, const char *stuff);

__END_DECLS

#endif //ABCDK_SHELL_DMI_H