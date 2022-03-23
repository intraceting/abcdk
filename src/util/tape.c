/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "util/tape.h"

static struct _abcdk_tape_density_dict
{
    uint8_t type;
    const char *msg;
} abcdk_tape_density_dict[] = {
    /* Copy from mt-st*/
    {0x00, "default"},
    {0x01, "NRZI (800 bpi)"},
    {0x02, "PE (1600 bpi)"},
    {0x03, "GCR (6250 bpi)"},
    {0x04, "QIC-11"},
    {0x05, "QIC-45/60 (GCR, 8000 bpi)"},
    {0x06, "PE (3200 bpi)"},
    {0x07, "IMFM (6400 bpi)"},
    {0x08, "GCR (8000 bpi)"},
    {0x09, "GCR (37871 bpi)"},
    {0x0a, "MFM (6667 bpi)"},
    {0x0b, "PE (1600 bpi)"},
    {0x0c, "GCR (12960 bpi)"},
    {0x0d, "GCR (25380 bpi)"},
    {0x0f, "QIC-120 (GCR 10000 bpi)"},
    {0x10, "QIC-150/250 (GCR 10000 bpi)"},
    {0x11, "QIC-320/525 (GCR 16000 bpi)"},
    {0x12, "QIC-1350 (RLL 51667 bpi)"},
    {0x13, "DDS (61000 bpi)"},
    {0x14, "EXB-8200 (RLL 43245 bpi)"},
    {0x15, "EXB-8500 or QIC-1000"},
    {0x16, "MFM 10000 bpi"},
    {0x17, "MFM 42500 bpi"},
    {0x18, "TZ86"},
    {0x19, "DLT 10GB"},
    {0x1a, "DLT 20GB"},
    {0x1b, "DLT 35GB"},
    {0x1c, "QIC-385M"},
    {0x1d, "QIC-410M"},
    {0x1e, "QIC-1000C"},
    {0x1f, "QIC-2100C"},
    {0x20, "QIC-6GB"},
    {0x21, "QIC-20GB"},
    {0x22, "QIC-2GB"},
    {0x23, "QIC-875"},
    {0x24, "DDS-2"},
    {0x25, "DDS-3"},
    {0x26, "DDS-4 or QIC-4GB"},
    {0x27, "Exabyte Mammoth"},
    {0x28, "Exabyte Mammoth-2"},
    {0x29, "QIC-3080MC"},
    {0x30, "AIT-1 or MLR3"},
    {0x31, "AIT-2"},
    {0x32, "AIT-3 or SLR7"},
    {0x33, "SLR6"},
    {0x34, "SLR100"},
    {0x40, "DLT1 40 GB, or Ultrium"},
    {0x41, "DLT 40GB, or Ultrium2"},
    {0x42, "LTO-2"},
    {0x44, "LTO-3"},
    {0x45, "QIC-3095-MC (TR-4)"},
    {0x46, "LTO-4"},
    {0x47, "DDS-5 or TR-5"},
    {0x48, "SDLT220"},
    {0x49, "SDLT320"},
    {0x4a, "SDLT600, T10000A"},
    {0x4b, "T10000B"},
    {0x4c, "T10000C"},
    {0x4d, "T10000D"},
    {0x51, "IBM 3592 J1A"},
    {0x52, "IBM 3592 E05"},
    {0x53, "IBM 3592 E06"},
    {0x54, "IBM 3592 E07"},
    {0x55, "IBM 3592 E08"},
    {0x58, "LTO-5"},
    {0x5a, "LTO-6"},
    {0x5c, "LTO-7"},
    {0x5d, "LTO-7-M8"},
    {0x5e, "LTO-8"},
    {0x71, "IBM 3592 J1A, encrypted"},
    {0x72, "IBM 3592 E05, encrypted"},
    {0x73, "IBM 3592 E06, encrypted"},
    {0x74, "IBM 3592 E07, encrypted"},
    {0x75, "IBM 3592 E08, encrypted"},
    {0x80, "DLT 15GB uncomp. or Ecrix"},
    {0x81, "DLT 15GB compressed"},
    {0x82, "DLT 20GB uncompressed"},
    {0x83, "DLT 20GB compressed"},
    {0x84, "DLT 35GB uncompressed"},
    {0x85, "DLT 35GB compressed"},
    {0x86, "DLT1 40 GB uncompressed"},
    {0x87, "DLT1 40 GB compressed"},
    {0x88, "DLT 40GB uncompressed"},
    {0x89, "DLT 40GB compressed"},
    {0x8c, "EXB-8505 compressed"},
    {0x90, "SDLT110 uncompr/EXB-8205 compr"},
    {0x91, "SDLT110 compressed"},
    {0x92, "SDLT160 uncompressed"},
    {0x93, "SDLT160 comprssed"}};

const char *abcdk_tape_density2string(uint8_t density)
{
    const char *msg_p = "Reserved";

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_tape_density_dict); i++)
    {
        if (abcdk_tape_density_dict[i].type != density)
            continue;

        msg_p = abcdk_tape_density_dict[i].msg;
        break;
    }

    return msg_p;
}

int abcdk_tape_operate(int fd, short cmd, int param, abcdk_scsi_io_stat *stat)
{
    struct mtop mp = {0};
    int chk;

    assert(fd >= 0);
    assert(stat != NULL);

    /*clear*/
    memset(stat, 0, sizeof(*stat));

    mp.mt_op = cmd;
    mp.mt_count = param;

    chk = ioctl(fd, MTIOCTOP, &mp);
    if (chk != 0)
        abcdk_scsi_request_sense(fd, 3000, stat);

    return chk;
}

int abcdk_tape_verify(int fd, uint32_t timeout, abcdk_scsi_io_stat *stat)
{
    uint8_t cdb[6] = {0};

    cdb[0] = 0x13;  /* 0x13 Verify  */
    cdb[1] |= 0x20; /*the VTE bit is set to one.*/

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_NONE, cdb, 6, NULL, 0, timeout, stat);
}

int abcdk_tape_seek(int fd, int cp, uint8_t part, uint64_t block,
                    uint32_t timeout, abcdk_scsi_io_stat *stat)
{
    uint8_t cdb[16] = {0};
    int chk;

    cdb[0] = 0x92;                /* 0x92 Locate  */
    cdb[1] |= (cp ? 0x02 : 0x00); /* 0x02 Change Partition */
    cdb[3] = part;
    ABCDK_PTR2OBJ(uint64_t, cdb, 4) = abcdk_endian_h_to_b64(block); /*4,5,6,7,8,9,10,11*/

    chk = abcdk_scsi_sgioctl2(fd, SG_DXFER_NONE, cdb, 16, NULL, 0, timeout, stat);

    return chk;
}

int abcdk_tape_tell(int fd, uint64_t *block, uint64_t *file, uint32_t *part,
                    uint32_t timeout, abcdk_scsi_io_stat *stat)
{
    uint8_t cdb[10] = {0};
    uint8_t buf[32] = {0};
    int chk;

    cdb[0] = 0x34; /* 0x34 Read Position   */
    cdb[1] = 0x06; /* 0x06 Long form */

    chk = abcdk_scsi_sgioctl2(fd, SG_DXFER_FROM_DEV, cdb, 10, buf, 32, timeout, stat);

    if (chk != 0 || stat->status != GOOD)
        return -1;

    if (block)
        *block = abcdk_endian_b_to_h64(ABCDK_PTR2U64(buf, 8)); /*8,9,10,11,12,13,14,15*/
    if (file)
        *file = abcdk_endian_b_to_h64(ABCDK_PTR2U64(buf, 16)); /*16,17,18,19,20,21,22,23*/
    if (part)
        *part = abcdk_endian_b_to_h32(ABCDK_PTR2U32(buf, 4)); /*4,5,6,7*/

    return 0;
}

abcdk_allocator_t *abcdk_tape_read_attribute(int fd, uint8_t part, uint16_t id,
                                             uint32_t timeout, abcdk_scsi_io_stat *stat)
{
    abcdk_allocator_t *alloc = NULL;
    uint16_t len = 0;

    uint8_t buf[255] = {0};
    uint8_t cdb[16] = {0};
    int chk;

    cdb[0] = 0x8C; /* 0x8C Read Attribute */
    cdb[1] = 0x00; /* 0x00 VALUE  */
    cdb[7] = part;
    ABCDK_PTR2U16(cdb, 8) = abcdk_endian_h_to_b16(id);           /*8,9*/
    ABCDK_PTR2U32(cdb, 10) = abcdk_endian_h_to_b32(sizeof(buf)); /*10,11,12,13*/

    chk = abcdk_scsi_sgioctl2(fd, SG_DXFER_FROM_DEV, cdb, 16, buf, sizeof(buf), timeout, stat);
    if (chk != 0 || stat->status != GOOD)
        return NULL;

    // printf("%u\n",abcdk_endian_b_to_h32(ABCDK_PTR2U32(buf, 0)));

    len = abcdk_endian_b_to_h16(ABCDK_PTR2U16(buf, 7)); /*7,8*/

    size_t sizes[5] = {sizeof(uint16_t), sizeof(uint8_t), sizeof(uint8_t), sizeof(uint16_t), len + 1};
    alloc = abcdk_allocator_alloc(sizes, 5, 0);
    if (!alloc)
        return NULL;

    ABCDK_PTR2U16(alloc->pptrs[ABCDK_TAPE_ATTR_ID], 0) = abcdk_endian_b_to_h16(ABCDK_PTR2U16(buf, 4)); /*4,5*/
    ABCDK_PTR2U8(alloc->pptrs[ABCDK_TAPE_ATTR_READONLY], 0) = (buf[6] >> 7);
    ABCDK_PTR2U8(alloc->pptrs[ABCDK_TAPE_ATTR_FORMAT], 0) = (buf[6] & 0x03);
    ABCDK_PTR2U16(alloc->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0) = abcdk_endian_b_to_h16(ABCDK_PTR2U16(buf, 7)); /*7,8*/
    memcpy(alloc->pptrs[ABCDK_TAPE_ATTR_VALUE], ABCDK_PTR2PTR(void, buf, 9), len);

    return alloc;
}

int abcdk_tape_write_attribute(int fd, uint8_t part, const abcdk_allocator_t *attr,
                               uint32_t timeout, abcdk_scsi_io_stat *stat)
{
    uint8_t buf[255] = {0};
    uint8_t cdb[16] = {0};
    uint8_t len = 0;

    assert(attr != NULL);

    len = 2 + 1 + 2 + ABCDK_PTR2U16(attr->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0);
    assert(4 + len <= 255);

    cdb[0] = 0x8D; /* 0x8D Write Attribute */
    cdb[1] = 0x01; /* 0x01 Write SYNC */
    cdb[7] = part;
    ABCDK_PTR2U32(cdb, 10) = abcdk_endian_h_to_b32(4 + len); /*10,11,12,13*/

    ABCDK_PTR2U32(buf, 0) = abcdk_endian_h_to_b32(len);
    ABCDK_PTR2U16(buf, 4) = abcdk_endian_h_to_b16(ABCDK_PTR2U16(attr->pptrs[ABCDK_TAPE_ATTR_ID], 0));
    ABCDK_PTR2U8(buf, 6) |= (ABCDK_PTR2U8(attr->pptrs[ABCDK_TAPE_ATTR_FORMAT], 0) & 0x03);
    ABCDK_PTR2U16(buf, 7) = abcdk_endian_h_to_b16(ABCDK_PTR2U16(attr->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0));
    memcpy(ABCDK_PTR2PTR(void, buf, 9), attr->pptrs[ABCDK_TAPE_ATTR_VALUE], ABCDK_PTR2U16(attr->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0));

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_TO_DEV, cdb, 16, buf, 4 + len, timeout, stat);
}
