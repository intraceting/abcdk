/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_SCSI_H
#define ABCDK_UTIL_SCSI_H

#include "abcdk/util/general.h"
#include "abcdk/util/endian.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS

/**
 * SCSI接口IO状态信息。
 */
typedef struct _abcdk_scsi_io_stat
{
    /**
     * 数据的未传输长度。
     */
    int32_t resid;

    /**
     * 感知信息。
     */
    uint8_t sense[255];

    /**
     * 感知信息的长度。
     */
    uint8_t senselen;

    /**
     * 状态。
     */
    uint8_t status;

    /**
     * 主机状态。
     */
    uint16_t host_status;

    /**
     * 设备状态。
     */
    uint16_t driver_status;

} abcdk_scsi_io_stat_t;

/**
 * 设备类型数字编码转字符串名称。
 *
 * @note 参考lsscsi命令。
 * 
 * @param type 类型。见TYPE_*。
 * @param longname !0 完整名称，0 缩写名称。
 * 
 */
const char *abcdk_scsi_type2string(uint8_t type, int longname);

/**
 * 通用的SCSI-V3指令接口。
 *
 * @param hdr 指令的指针。 in sg.h
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_scsi_sgioctl(int fd, struct sg_io_hdr *hdr);

/**
 * 通用的SCSI-V3指令接口。
 *
 * @param timeout 超时(毫秒)。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_scsi_sgioctl2(int fd, int direction,
                        uint8_t *cdb, uint8_t cdblen,
                        uint8_t *transfer, uint32_t transferlen,
                        uint32_t timeout, abcdk_scsi_io_stat_t *stat);

/**
 * 获取sense信息KEY字段值。
 */
uint8_t abcdk_scsi_sense_key(uint8_t *sense);

/**
 * 获取sense信息ASC字段值。
 */
uint8_t abcdk_scsi_sense_code(uint8_t *sense);

/**
 * 获取sense信息ASCQ字段值。
 */
uint8_t abcdk_scsi_sense_qualifier(uint8_t *sense);

/**
 * 测试设备是否准备好。
 *
 * 0x00H
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_scsi_test(int fd, uint32_t timeout, abcdk_scsi_io_stat_t *stat);

/**
 * 查询状态信息。
 *
 * cdb = 0x03
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_scsi_request_sense(int fd, uint32_t timeout, abcdk_scsi_io_stat_t *stat);

/**
 * 查询设备信息。
 *
 * cdb = 0x12H
 *
 * @param vpd 是否查询虚拟页面。0 否，!0 是。
 * @param pcd 页面ID。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_scsi_inquiry(int fd, int vpd, uint8_t pcd,
                       uint8_t *transfer, uint32_t transferlen,
                       uint32_t timeout, abcdk_scsi_io_stat_t *stat);

/**
 * 获取设备标准信息。
 *
 * cdb = 0x12, VPD = 0, PCD = 0x00
 *
 * @param type 返回类型，可以为NULL(0)。
 * @param vendor 返回生产商，可以为NULL(0)。
 * @param product 返回产品名称，可以为NULL(0)。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_scsi_inquiry_standard(int fd, uint8_t *type, char vendor[8], char product[16],
                                uint32_t timeout, abcdk_scsi_io_stat_t *stat);

/**
 * 获取设备SN信息。
 *
 * cdb = 0x12, VPD = 1, PCD = 0x80
 *
 * @param type 返回类型，可以为NULL(0)。
 * @param serial 返回SN码，可以为NULL(0)。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_scsi_inquiry_serial(int fd, uint8_t *type, char serial[255],
                              uint32_t timeout, abcdk_scsi_io_stat_t *stat);

__END_DECLS

#endif // ABCDK_UTIL_SCSI_H