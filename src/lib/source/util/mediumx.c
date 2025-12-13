/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/mediumx.h"

static struct _abcdk_mediumx_sense_dict
{   
    uint8_t key;
    uint8_t asc;
    uint8_t ascq;
    const char *msg;
}abcdk_mediumx_sense_dict[] = {
    /*KEY=0x00*/
    {0x00, 0x00, 0x00, "No Sense"},
    /*KEY=0x01*/
    {0x01, 0x00, 0x00, "Recovered Error"},
    /*KEY=0x02*/
    {0x02, 0x00, 0x00, "Not Ready"},
    /*KEY=0x03*/
    {0x03, 0x00, 0x00, "Medium Error"},
    /*KEY=0x04*/
    {0x04, 0x00, 0x00, "Hardware Error"},
    /*KEY=0x05*/
    {0x05, 0x00, 0x00, "Illegal Request"},
    {0x05, 0x21, 0x01, "无效的地址"},
    {0x05, 0x24, 0x00, "无效的地址或超出范围"},
    {0x05, 0x3b, 0x0d, "目标地址有介质"},
    {0x05, 0x3b, 0x0e, "源地址无介质"},
    {0x05, 0x53, 0x02, "Library media removal prevented state set"},
    {0x05, 0x53, 0x03, "Drive media removal prevented state set"},
    {0x05, 0x44, 0x80, "Bad status library controller"},
    {0x05, 0x44, 0x81, "Source not ready"},
    {0x05, 0x44, 0x82, "Destination not ready"},
    /*KEY=0x06*/
    {0x06, 0x00, 0x00, "Unit Attention"},
    /*KEY=0x0b*/
    {0x0b, 0x00, 0x00, "Command Aborted"}
};

const char *abcdk_mediumx_sense2string(uint8_t key, uint8_t asc , uint8_t ascq)
{
    const char *msg_p = NULL;

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_mediumx_sense_dict); i++)
    {
        if (abcdk_mediumx_sense_dict[i].key != key)
            continue;

        msg_p = abcdk_mediumx_sense_dict[i].msg;

        if (abcdk_mediumx_sense_dict[i].asc != asc || abcdk_mediumx_sense_dict[i].ascq != ascq)
            continue;

        msg_p = abcdk_mediumx_sense_dict[i].msg;
        break;
    }

    return msg_p;
}

void abcdk_mediumx_stat_dump(FILE *fp,abcdk_scsi_io_stat_t *stat)
{
    uint8_t key = 0, asc = 0, ascq = 0;
    const char *msg_p = NULL;

    key = abcdk_scsi_sense_key(stat->sense);
    asc = abcdk_scsi_sense_code(stat->sense);
    ascq = abcdk_scsi_sense_qualifier(stat->sense);

    msg_p = abcdk_mediumx_sense2string(key, asc, ascq);

    fprintf(stderr, "Sense(KEY=%02X,ASC=%02X,ASCQ=%02X): %s.\n", key, asc, ascq, (msg_p ? msg_p : "Unknown"));
}

int abcdk_mediumx_inventory(int fd, uint16_t address, uint16_t count,
                            uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{

    uint8_t cdb[10] = {0};

    cdb[0] = 0x37; /* 0x37 or E7 Initialize Element Status With Range */
    ABCDK_PTR2U8(cdb, 1) |= (count > 0 ? 0x01 : 0);
    ABCDK_PTR2U16(cdb, 2) = abcdk_endian_h_to_b16(address);
    ABCDK_PTR2U16(cdb, 6) = abcdk_endian_h_to_b16(count);

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_NONE, cdb, 10, NULL, 0, timeout, stat);
}

int abcdk_mediumx_move_medium(int fd, uint16_t t, uint16_t src, uint16_t dst,
                              uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{

    uint8_t cdb[12] = {0};

    cdb[0] = 0xA5; /* 0xA5 Move Medium code */
    ABCDK_PTR2U16(cdb, 2) = abcdk_endian_h_to_b16(t);
    ABCDK_PTR2U16(cdb, 4) = abcdk_endian_h_to_b16(src);
    ABCDK_PTR2U16(cdb, 6) = abcdk_endian_h_to_b16(dst);

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_NONE, cdb, 12, NULL, 0, timeout, stat);
}

int abcdk_mediumx_prevent_medium_removal(int fd, int disable,
                                         uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    uint8_t cdb[6] = {0};

    cdb[0] = 0x1E; /* 0x1E Prevent Allow Medium Removal  */
    cdb[4] = disable ? 0x01 : 0x00;

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_NONE, cdb, 6, NULL, 0, timeout, stat);
}

int abcdk_mediumx_mode_sense(int fd, uint8_t pctrl, uint8_t pcode, uint8_t spcode,
                             uint8_t *transfer, uint8_t transferlen,
                             uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    uint8_t cdb[6] = {0};

    cdb[0] = 0x1A; /* 0x1A Mode Sense  */
    cdb[1] = 0x08;
    cdb[2] = (pctrl << 6) | (pcode & 0x3F);
    cdb[3] = spcode;
    cdb[4] = transferlen;

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_FROM_DEV, cdb, 6, transfer, transferlen, timeout, stat);
}

int abcdk_mediumx_read_element_status(int fd, uint8_t type,
                                      int voltag, int dvcid,
                                      uint16_t address, uint16_t count,
                                      uint8_t *transfer, uint32_t transferlen,
                                      uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    uint8_t cdb[12] = {0};

    cdb[0] = 0xB8;                                          /* 0xB8 Read Element Status */
    cdb[1] = (voltag ? 0x10 : 0x00) | (type & 0x0F);        /*VolTag and type*/
    ABCDK_PTR2U16(cdb, 2) = abcdk_endian_h_to_b16(address); /*2,3*/
    ABCDK_PTR2U16(cdb, 4) = abcdk_endian_h_to_b16(count);   /*4,5*/
    cdb[6] = (dvcid ? 0x01 : 0);                            /*DVCID*/
    abcdk_endian_h_to_b24(cdb + 7, transferlen);            /*7,8,9*/

    return abcdk_scsi_sgioctl2(fd, SG_DXFER_FROM_DEV, cdb, 12, transfer, transferlen, timeout, stat);
}

void abcdk_mediumx_parse_element_status(abcdk_tree_t *list, const uint8_t *element, uint16_t count)
{
    assert(list != NULL && element != NULL && count > 0);

    /**/
    uint8_t type = element[8];
    int pvoltag = (element[9] & 0x80) ? 1 : 0;
    int avoltag = (element[9] & 0x40) ? 1 : 0;
    uint16_t psize = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, element, 10));
    const uint8_t *ptr = element + 16; /*First Page*/

    for (uint16_t i = 0; i < count; i++)
    {
        /*申请节点.*/
        size_t sizes[5] = {sizeof(uint16_t), sizeof(uint8_t), sizeof(uint8_t), 36 + 1, 32 + 1};
        abcdk_tree_t *one = abcdk_tree_alloc2(sizes, 5, 0);

        /*如果节点申请失败提结束.*/
        if (one == NULL)
            break;

        /*是否有条码字段.*/
        uint8_t volsize = (pvoltag ? 36 : 0);

        /*获取部分字段.*/
        ABCDK_PTR2U16(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_ADDR], 0) = abcdk_endian_b_to_h16(ABCDK_PTR2U16(ptr, 0));
        ABCDK_PTR2U8(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_TYPE], 0) = type;
        ABCDK_PTR2U8(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_ISFULL], 0) = ((ptr[2] & 0x01) ? 1 : 0);
        if (volsize > 0)
            memcpy(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_BARCODE], ptr + 12, volsize);

        /*是否有机械臂或驱动器信息.*/
        uint8_t dvcid_set = ptr[12 + volsize] & 0x0F;

        /*0x01或0x02有效.*/
        if (dvcid_set == 0)
            goto next;

        /*机械臂或驱动器才有下面的数据.*/
        if (ABCDK_MEDIUMX_ELEMENT_CHANGER == type || ABCDK_MEDIUMX_ELEMENT_DXFER == type)
        {
            uint8_t dvcid_type = ptr[13 + volsize] & 0x0F;
            uint8_t dvcid_length = ptr[15 + volsize];
            /**/
            if (dvcid_type == 0x00)
            {
                /*Only Serial Number.*/
                memcpy(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_DVCID], ptr + 16 + volsize, dvcid_length);
            }
            else if (dvcid_type == 0x01)
            {
                if (dvcid_length == 0x0A || dvcid_length == 0x20)
                {
                    /*
                     * Only Serial Number.
                     *
                     * Is Spectra Tape Libraries?
                     */
                    memcpy(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_DVCID], ptr + 16 + volsize, dvcid_length);
                }
                else
                {
                    /*
                     * type == 0x01, which is equivalent to the drive's Inquiry page 83h.
                     *
                     * VENDOR(8)+PRODUCT(16)+SERIAL(10)
                     */
                    memcpy(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_DVCID], ptr + 16 + volsize + 8 + 16, 10);
                }
            }
            else
            {
                /*0x02~0x0f.*/;
            }
        }

    next:

        /*清除两端的空格.*/
        abcdk_strtrim(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_BARCODE], isspace, 2);
        abcdk_strtrim(one->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_DVCID], isspace, 2);

        /*添加到子节点末尾.*/
        abcdk_tree_insert2(list, one, 0);

        /*下一页.*/
        ptr += psize;
    }
}

int abcdk_mediumx_inquiry_element_status(abcdk_tree_t *list, int fd, int voltag, int dvcid,
                                         uint32_t timeout, abcdk_scsi_io_stat_t *stat)
{
    char buf[255] = {0};
    int buf2size = 0,buf2max = 0;
    uint8_t *buf2 = NULL;
    uint16_t addr[4] = {0};
    uint16_t count[4] = {0};
    int chk;

    chk = abcdk_mediumx_mode_sense(fd, 0, 0x1d, 0, buf, 255, timeout, stat);
    if (chk != 0)
        return -1;

    /*15MB MAX!!!*/
    buf2size = 0x00ffffff; 
    buf2 = (uint8_t *)abcdk_heap_alloc(buf2size);
    if (!buf2)
        return -1;

    /*ABCDK_MEDIUMX_ELEMENT_CHANGER:4+2,4+4*/
    addr[0] = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, buf, 4 + 2));
    count[0] = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, buf, 4 + 4));
    
    chk = abcdk_mediumx_read_element_status(fd, ABCDK_MEDIUMX_ELEMENT_CHANGER, voltag, dvcid,
                                            addr[0],count[0],
                                            buf2, 0xA000, -1, stat);
    if (chk != 0)
        goto final;

    abcdk_mediumx_parse_element_status(list, buf2, count[0]);

    /*ABCDK_MEDIUMX_ELEMENT_STORAGE:4+6,4+8*/
    addr[1] = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, buf, 4 + 6));
    count[1] = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, buf, 4 + 8));

    /* Page Header(8)+ Page Status(8)+ Elements(N)*52 */
    buf2max = ABCDK_MIN(8 + 8 + count[1] * 52, buf2size);
    chk = abcdk_mediumx_read_element_status(fd, ABCDK_MEDIUMX_ELEMENT_STORAGE, voltag, dvcid,
                                            addr[1],count[1],
                                            buf2, buf2max, -1, stat);
    if (chk != 0)
        goto final;

    abcdk_mediumx_parse_element_status(list, buf2, count[1]);

    
    /*ABCDK_MEDIUMX_ELEMENT_IE_PORT:4+10,4+12*/
    addr[2] = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, buf, 4 + 10));
    count[2] = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, buf, 4 + 12));

    chk = abcdk_mediumx_read_element_status(fd, ABCDK_MEDIUMX_ELEMENT_IE_PORT, voltag, dvcid,
                                            addr[2],count[2],
                                            buf2, 0xA000, -1, stat);
    if (chk != 0)
        goto final;

    abcdk_mediumx_parse_element_status(list, buf2, count[2]);

    /*ABCDK_MEDIUMX_ELEMENT_DXFER:4+14,4+16*/
    addr[3] = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, buf, 4 + 14));
    count[3] = abcdk_endian_b_to_h16(ABCDK_PTR2OBJ(uint16_t, buf, 4 + 16));

    chk = abcdk_mediumx_read_element_status(fd, ABCDK_MEDIUMX_ELEMENT_DXFER, voltag, dvcid,
                                            addr[3],count[3],
                                            buf2, 0xA000, -1, stat);
    if (chk != 0)
        goto final;

    abcdk_mediumx_parse_element_status(list, buf2, count[3]);

final:

    abcdk_heap_freep((void **)&buf2);

    return chk;
}

typedef struct _abcdk_mediumx_find_param
{
    uint8_t find_type;

    uint16_t changer_addr;

}abcdk_mediumx_find_param_t;

int _abcdk_mediumx_find_changer_address_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_mediumx_find_param_t *param_p = (abcdk_mediumx_find_param_t *)opaque;

    /*已经结束.*/
    if(depth == SIZE_MAX)
        return -1;

    if (depth == 0)
        return 1;

    if (ABCDK_PTR2U8(node->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_TYPE], 0) != param_p->find_type)
        return 1;

    param_p->changer_addr = ABCDK_PTR2U16(node->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_ADDR], 0);

    return -1;
}

uint16_t abcdk_mediumx_find_changer_address(abcdk_tree_t *list)
{
    abcdk_mediumx_find_param_t param = {0};

    assert(list != NULL);

    param.find_type = ABCDK_MEDIUMX_ELEMENT_CHANGER;

    abcdk_tree_iterator_t it = {0, &param, _abcdk_mediumx_find_changer_address_cb};
    abcdk_tree_scan(list, &it);

    return param.changer_addr;
}