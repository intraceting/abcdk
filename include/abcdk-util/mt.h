/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_MT_H
#define ABCDK_UTIL_MT_H

#include "abcdk-util/general.h"
#include "abcdk-util/allocator.h"
#include "abcdk-util/scsi.h"

__BEGIN_DECLS

/**
 * 属性的字段。
*/
enum _abcdk_mt_attr_field
{
    /** ID 字段索引.*/
    ABCDK_MT_ATTR_ID = 0,
#define ABCDK_MT_ATTR_ID ABCDK_MT_ATTR_ID

    /** 只读 字段索引.*/
    ABCDK_MT_ATTR_READONLY = 1,
#define ABCDK_MT_ATTR_READONLY ABCDK_MT_ATTR_READONLY

    /** 格式 字段索引. 
     *  
     *  00b BINARY.
     *  01b ASCII The ATTRIBUTE VALUE field contains left-aligned ASCII data.
     *  10b TEXT The attribute contains textual data. 
     *  11b Reserved.
    */
    ABCDK_MT_ATTR_FORMAT = 2,
#define ABCDK_MT_ATTR_FORMAT ABCDK_MT_ATTR_FORMAT

    /** 长度 字段索引. */
    ABCDK_MT_ATTR_LENGTH = 3,
#define ABCDK_MT_ATTR_LENGTH ABCDK_MT_ATTR_LENGTH

    /** 值 字段索引. */
    ABCDK_MT_ATTR_VALUE = 4
#define ABCDK_MT_ATTR_VALUE ABCDK_MT_ATTR_VALUE

};

/**
 * 磁带型号数字编码转字符串编码。
 * 
 * 参考mt命令。
 * 
 * @return 字符串的指针。
 */
const char *abcdk_mt_density2string(uint8_t density);

/**
 * 磁带操作。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_mt_operate(int fd,short cmd,int param);

/**压缩功能开关。*/
#define abcdk_mt_compression(fd, enable)     abcdk_mt_operate(fd, MTCOMPRESSION, enable)

/**设置读写块大小。*/
#define abcdk_mt_blocksize(fd, blocks)       abcdk_mt_operate(fd, MTSETBLK, blocks)

/**写FILEMARK标记。*/
#define abcdk_mt_writefm(fd, counts)         abcdk_mt_operate(fd, MTWEOF, counts)

/**倒带。*/
#define abcdk_mt_rewind(fd, offline)         abcdk_mt_operate(fd, offline ? MTOFFL : MTREW, 0)

/**加载磁带。*/
#define abcdk_mt_load(fd)                    abcdk_mt_operate(fd, MTLOAD, 0)

/**卸载磁带*/
#define abcdk_mt_unload(fd)                  abcdk_mt_operate(fd, MTUNLOAD, 0)

/**禁止卸载磁带。*/
#define abcdk_mt_lock(fd)                    abcdk_mt_operate(fd, MTLOCK, 0)

/**允许卸载磁带。*/
#define abcdk_mt_unlock(fd)                  abcdk_mt_operate(fd, MTUNLOCK, 0)

/**擦除磁带数据。*/
#define abcdk_mt_erase(fd, physical)         abcdk_mt_operate(fd, MTERASE, physical) /*dangerous*/

/**改变活动分区。*/
#define abcdk_mt_part(part)                  abcdk_mt_operate(fd, MTSETPART, part);

/**
 * 磁带较验。
 * 
 * cdb = 0x13
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_mt_verify(int fd,uint32_t timeout, abcdk_scsi_io_stat *stat);

/**
 * 定位磁头的位置。
 * 
 * cdb = 0x92
 * 
 * @param cp 是否改变当前活动分区。
 * @param part 分区号。
 * @param block 逻辑块索引。
 * 
 * @return 0 成功，-1 失败。
 * 
 * @warning  New tape(KEY = 0x08,ASC = 0x14,ASCQ = 0x03).
 * @warning  End of data (KEY = 0x08,ASC = 0x00,ASCQ = 0x05).
*/
int abcdk_mt_seek(int fd, int cp, uint8_t part, uint64_t block,
                  uint32_t timeout, abcdk_scsi_io_stat *stat);

/**
 * 读取磁头当前位置。
 * 
 * cdb = 0x34
 * 
 * @param block 返回逻辑块索引，为NULL(0)忽略。
 * @param file 返回逻辑文件的索引，为NULL(0)忽略。
 * @param part 返回分区号，为NULL(0)忽略。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_mt_tell(int fd, uint64_t *block, uint64_t *file, uint32_t *part,
                  uint32_t timeout, abcdk_scsi_io_stat *stat);

/*
 * See "Medium auxiliary memory attributes (MAM)" from the manufacturer's SCSI manual.
 * ______________________________________________________________________________________________
 * |Attribute Identifier|Name                               |Attribute Length(in bytes)|Format  |
 * ----------------------------------------------------------------------------------------------
 * |0000h               |REMAINING CAPACITY IN PARTITION    |8                         |BINARY  |
 * |0001h               |MAXIMUM CAPACITY IN PARTITION      |8                         |BINARY  |
 * |0002h               |TAPEALERT FLAGS                    |8                         |BINARY  |
 * |0003h               |LOAD COUNT                         |8                         |BINARY  |
 * |0004h               |MAM SPACE REMAINING                |8                         |BINARY  |
 * |0005h               |ASSIGNING ORGANIZATION             |8                         |ASCII   |
 * |0006h               |FORMATTED DENSITY CODE             |1                         |BINARY  |
 * |0007h               |INITIALIZATION COUNT               |2                         |BINARY  |
 * 
 * ----------------------------------------------------------------------------------------------
 * |0220h               |TOTAL MBYTES WRITTEN IN MEDIUM LIFE|8                         |BINARY  |
 * |0221h               |TOTAL MBYTES READ IN MEDIUM LIFE   |8                         |BINARY  |
 * |0222h               |TOTAL MBYTES WRITTEN IN CURRENT    |8                         |BINARY  |
 * |0223h               |TOTAL MBYTES READ IN CURRENT       |8                         |BINARY  |
 * 
 * ----------------------------------------------------------------------------------------------
 * |0400h               |MEDIUM MANUFACTURER                |8                         |ASCII   |
 * |0401h               |MEDIUM SERIAL NUMBER               |32                        |ASCII   |
 * |0402h               |MEDIUM LENGTH                      |4                         |BINARY  |
 * |0403h               |MEDIUM WIDTH                       |4                         |BINARY  |
 * |0404h               |ASSIGNING ORGANIZATION             |8                         |ASCII   |
 * |0405h               |MEDIUM DENSITY CODE                |1                         |BINARY  |
 * |0406h               |MEDIUM MANUFACTURE DATE            |8                         |ASCII   |
 * |0407h               |MAM CAPACITY                       |8                         |BINARY  |
 * |0408h               |MEDIUM TYPE                        |1                         |BINARY  |
 * |0409h               |MEDIUM TYPE INFORMATION            |2                         |BINARY  |
 * 
 * ----------------------------------------------------------------------------------------------
 * |0800h               |APPLICATION VENDOR                 |8                         |ASCII   |
 * |0801h               |APPLICATION NAME                   |32                        |ASCII   |
 * |0802h               |APPLICATION VERSION                |8                         |ASCII   |
 * |0803h               |USER MEDIUM TEXT LABEL             |160                       |TEXT    |
 * |0804h               |DATE AND TIME LAST WRITTEN         |12                        |ASCII   |
 * |0805h               |TEXT LOCALIZATION IDENTIFIER       |1                         |BINARY  |
 * |0806h               |BARCODE                            |32                        |ASCII   |
 * |0807h               |OWNING HOST TEXTUAL NAME           |80                        |TEXT    |
 * |0808h               |MEDIA POOL                         |160                       |TEXT    |
 * |0809h               |MEDIUM TYPE INFORMATION            |2                         |BINARY  |
 * |080Bh               |APPLICATION FORMAT VERSION         |16                        |ASCII   |
 * ----------------------------------------------------------------------------------------------
 */

/** 
 * 读取磁带的属性。
 * 
 * 如果属性值的是二进制数据，并且也是整型数据时，以网络字节序存储。
 * 
 * cdb = 0x8C
 * 
 * @param part 分区号。
 * @param id 字段ID。
 * 
 * @return !NULL(0) 成功(属性的指针)，NULL(0) 失败。
 * 
 * @warning 磁带物理只读锁不影响此功能。
*/
abcdk_allocator_t *abcdk_mt_read_attribute(int fd, uint8_t part, uint16_t id,
                                         uint32_t timeout, abcdk_scsi_io_stat *stat);

/** 
 * 写入磁带的属性。
 * 
 * 如果属性值的是二进制数据，并且也是整型数据时，以网络字节序存储。
 * 
 * cdb = 0x8D
 * 
 * @return 0 成功，-1 失败。
 * 
 * @warning 磁带物理只读锁不影响此功能。
*/
int abcdk_mt_write_attribute(int fd, uint8_t part, const abcdk_allocator_t *attr,
                            uint32_t timeout, abcdk_scsi_io_stat *stat);

__END_DECLS

#endif //ABCDK_UTIL_MT_H