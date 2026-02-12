/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/scsi.h"

const char *abcdk_scsi_type2string(uint8_t type, int longname)
{
    static const char *types_table[] =
        {
            "Direct-Access",
            "Sequential-Access",
            "Printer",
            "Processor",
            "Write-once",
            "CD-ROM",
            "Scanner",
            "Optical memory",
            "Medium Changer",
            "Communications",
            "Unknown (0xa)",
            "Unknown (0xb)",
            "Storage array",
            "Enclosure",
            "Simplified direct-access",
            "Optical card read/writer",
            "Bridge controller",
            "Object based storage",
            "Automation Drive interface",
            "Reserved (0x13)",
            "Reserved (0x14)",
            "Reserved (0x15)",
            "Reserved (0x16)",
            "Reserved (0x17)",
            "Reserved (0x18)",
            "Reserved (0x19)",
            "Reserved (0x1a)",
            "Reserved (0x1b)",
            "Reserved (0x1c)",
            "Reserved (0x1e)",
            "Well known LU",
            "No device",
        };

    static const char *short_types_table[] =
        {
            "disk",
            "tape",
            "printer",
            "process",
            "worm",
            "cd/dvd",
            "scanner",
            "optical",
            "mediumx",
            "comms",
            "(0xa)",
            "(0xb)",
            "storage",
            "enclosu",
            "sim dsk",
            "opti rd",
            "bridge",
            "osd",
            "adi",
            "(0x13)",
            "(0x14)",
            "(0x15)",
            "(0x16)",
            "(0x17)",
            "(0x18)",
            "(0x19)",
            "(0x1a)",
            "(0x1b)",
            "(0x1c)",
            "(0x1e)",
            "wlun",
            "no dev",
        };

    /*已知的定义.*/
    type = (0x1f &type);

    return (longname?types_table[type]:short_types_table[type]);
}

int abcdk_scsi_sgioctl(int fd, struct sg_io_hdr *hdr)
{
    assert(fd >= 0 && hdr != NULL);

    assert(hdr->interface_id == 'S');
    assert(hdr->dxfer_direction == SG_DXFER_NONE || hdr->dxfer_direction == SG_DXFER_TO_DEV ||
           hdr->dxfer_direction == SG_DXFER_FROM_DEV || hdr->dxfer_direction == SG_DXFER_TO_FROM_DEV);
    assert(hdr->cmdp != NULL && hdr->cmd_len > 0);
    assert((hdr->dxferp != NULL && hdr->dxfer_len > 0) || (hdr->dxferp == NULL && hdr->dxfer_len <= 0));
    assert((hdr->sbp != NULL && hdr->mx_sb_len > 0) || (hdr->sbp == NULL && hdr->mx_sb_len <= 0));

    return ioctl(fd, SG_IO, hdr);
}

int abcdk_scsi_sgioctl2(int fd, int direction,
                        uint8_t *cdb, uint8_t cdblen,
                        uint8_t *transfer, uint32_t transferlen,
                        uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    struct sg_io_hdr hdr = {0};
    int chk;

    assert(fd >= 0 && cdb != NULL && cdblen > 0);
    assert((transfer != NULL && transferlen > 0) ||
           (transfer == NULL && transferlen <= 0));
    assert(stat != NULL);

    /*clear*/
    memset(stat, 0, sizeof(*stat));

    hdr.interface_id = 'S';
    hdr.dxfer_direction = direction;
    hdr.cmdp = cdb;
    hdr.cmd_len = cdblen;
    hdr.dxferp = transfer;
    hdr.dxfer_len = transferlen;
    hdr.sbp = stat->sense;
    hdr.mx_sb_len = sizeof(stat->sense);
    hdr.timeout = timeout;

    chk = abcdk_scsi_sgioctl(fd, &hdr);
    if (chk != 0)
        return -1;

    stat->status = hdr.status;
    stat->host_status = hdr.host_status;
    stat->driver_status = hdr.driver_status;
    stat->senselen = hdr.sb_len_wr;
    stat->resid = hdr.resid;

    return 0;
}

uint8_t abcdk_scsi_sense_key(uint8_t *sense)
{
    assert(sense != NULL);

    return sense[2] & 0xf;
}

uint8_t abcdk_scsi_sense_code(uint8_t *sense)
{
    assert(sense != NULL);

    return sense[12];
}

uint8_t abcdk_scsi_sense_qualifier(uint8_t *sense)
{
    assert(sense != NULL);

    return sense[13];
}

int abcdk_scsi_test(int fd, uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    uint8_t cdb[6] = {0};

    cdb[0] = 0x00; /*00H is TEST UNIT READY*/

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_NONE, cdb, 6, NULL, 0, timeout, stat);
}

int abcdk_scsi_request_sense(int fd, uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    uint8_t cdb[6] = {0};

    cdb[0] = 0x03; /*03H is Request Sense*/
    cdb[4] = sizeof(stat->sense);

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_NONE, cdb, 6, NULL, 0, timeout, stat);
}

int abcdk_scsi_inquiry(int fd, int vpd, uint8_t vid,
                       uint8_t *transfer, uint32_t transferlen,
                       uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    uint8_t cdb[6] = {0};

    cdb[0] = 0x12;                /*12H is INQUIRY*/
    cdb[1] = (vpd ? 0x01 : 0x00); /* Enable Vital Product Data */
    cdb[2] = (vpd ? vid : 0x00);  /* Return PAGE CODE */
    cdb[4] = transferlen;

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_FROM_DEV, cdb, 6, transfer, transferlen, timeout, stat);
}

int abcdk_scsi_inquiry_standard(int fd, uint8_t *type, char vendor[8], char product[16],
                                uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    uint8_t tmp[255] = {0};
    int chk;

    chk = abcdk_scsi_inquiry(fd, 0, 0x00, tmp, 255, timeout, stat);
    if (chk != 0)
        return -1;

    if (stat->status != GOOD)
        return -1;

    /* TYPE, VENDOR, PRODUCT.*/
    if (type)
        *type = tmp[0] & 0x1f;
    if (vendor)
        memcpy(vendor, tmp + 8, 8);
    if (product)
        memcpy(product, tmp + 16, 16);

    /* 去掉两端的空格. */
    if (vendor)
        abcdk_strtrim(vendor, isspace, 2);
    if (product)
        abcdk_strtrim(product, isspace, 2);

    return 0;
}

int abcdk_scsi_inquiry_serial(int fd, uint8_t *type, char serial[255],
                              uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    uint8_t tmp[255] = {0};
    int chk;

    chk = abcdk_scsi_inquiry(fd, 1, 0x80, tmp, 255, timeout, stat);
    if (chk != 0)
        return -1;

    if (stat->status != GOOD)
        return -1;

    /* TYPE, SERIAL.*/
    if (type)
        *type = tmp[0] & 0x1f;
    if (serial)
        memcpy(serial, tmp + 4, tmp[3]);

    /* 去掉两端的空格. */
    if (serial)
        abcdk_strtrim(serial, isspace, 2);

    return 0;
}
