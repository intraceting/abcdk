/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "abcdkutil/general.h"
#include "abcdkutil/getargs.h"
#include "abcdkutil/scsi.h"
#include "abcdkutil/mt.h"

/**/
enum _abcdkmt_cmd
{
    /** 查询驱动器信息。*/
    ABCDKMT_HWINFO = 1,
#define ABCDKMT_HWINFO ABCDKMT_HWINFO

    /** 查询磁带状态。*/
    ABCDKMT_STATUS = 2,
#define ABCDKMT_STATUS ABCDKMT_STATUS

    /** 倒带。*/
    ABCDKMT_REWIND = 3,
#define ABCDKMT_REWIND ABCDKMT_REWIND

    /** 加载。*/
    ABCDKMT_LOAD = 4,
#define ABCDKMT_LOAD ABCDKMT_LOAD

    /** 卸载。*/
    ABCDKMT_UNLOAD = 5,
#define ABCDKMT_UNLOAD ABCDKMT_UNLOAD

    /** 加锁。*/
    ABCDKMT_LOCK = 6,
#define ABCDKMT_LOCK ABCDKMT_LOCK

    /** 解锁。*/
    ABCDKMT_UNLOCK = 7,
#define ABCDKMT_UNLOCK ABCDKMT_UNLOCK

    /** 读取磁头位置(逻辑)。*/
    ABCDKMT_READ_POS = 8,
#define ABCDKMT_READ_POS ABCDKMT_READ_POS

    /** 移动磁头位置(逻辑)。*/
    ABCDKMT_SEEK_POS = 9,
#define ABCDKMT_SEEK_POS ABCDKMT_SEEK_POS

    /** 写FILEMARK。*/
    ABCDKMT_WRITE_FILEMARK = 10
#define ABCDKMT_WRITE_FILEMARK ABCDKMT_WRITE_FILEMARK
};

void _abcdkmt_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    /*Clear errno.*/
    errno = 0;

    abcdk_proc_basename(name);

#ifdef BUILD_VERSION_DATETIME
    fprintf(stderr, "\n%s Build %s\n", name, BUILD_VERSION_DATETIME);
#endif //BUILD_VERSION_DATETIME

    fprintf(stderr, "\n%s Version %d.%d\n", name, ABCDK_VERSION_MAJOR, ABCDK_VERSION_MINOR);

    if (only_version)
        return;

    fprintf(stderr, "\nSYNOPSIS:\n");
        
    fprintf(stderr, "\n%s [ --dev < FILE > ] [ OPTIONS ] \n",name);

    fprintf(stderr, "\nOPTIONS:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\tShow this help message and exit.\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\tOutput version information and exit.\n");

    fprintf(stderr, "\n\t--dev < FILE >\n");
    fprintf(stderr, "\t\tBlock SCSI device.\n");

    fprintf(stderr, "\n\t--pos-block < NUMBER >\n");
    fprintf(stderr, "\t\tLogical object identifier.\n");

    fprintf(stderr, "\n\t--pos-part < NUMBER >\n");
    fprintf(stderr, "\t\tPartition identifier. default: 0\n");

    fprintf(stderr, "\n\t--filemarks < NUMBER >\n");
    fprintf(stderr, "\t\tLogical object numbers. default: 1\n");

    fprintf(stderr, "\n\t--cmd < NUMBER >\n");
    fprintf(stderr, "\t\tCommand. default: %d\n", ABCDKMT_STATUS);

    fprintf(stderr, "\n\t\t%d: Report driver baseinfo.\n", ABCDKMT_HWINFO);
    fprintf(stderr, "\t\t%d: Report medium baseinfo.\n", ABCDKMT_STATUS);
    fprintf(stderr, "\t\t%d: Rewind.\n", ABCDKMT_REWIND);
    fprintf(stderr, "\t\t%d: Load.\n", ABCDKMT_LOAD);
    fprintf(stderr, "\t\t%d: Unload.\n", ABCDKMT_UNLOAD);
    fprintf(stderr, "\t\t%d: Lock.\n", ABCDKMT_LOCK);
    fprintf(stderr, "\t\t%d: Unlock.\n", ABCDKMT_UNLOCK);
    fprintf(stderr, "\t\t%d: Read position.\n", ABCDKMT_READ_POS);
    fprintf(stderr, "\t\t%d: Seek position.\n", ABCDKMT_SEEK_POS);
    fprintf(stderr, "\t\t%d: Write filemark.\n", ABCDKMT_WRITE_FILEMARK);
}

static struct _abcdkmt_sense_dict
{   
    uint8_t key;
    uint8_t asc;
    uint8_t ascq;
    const char *msg;
}abcdkmt_sense_dict[] = {
    /*KEY=0x00*/
    {0x00, 0x00, 0x00, "No Sense"},
    /*KEY=0x01*/
    {0x01, 0x00, 0x00, "Recovered Error"},
    /*KEY=0x02*/
    {0x02, 0x00, 0x00, "Not Ready"},
    {0x02, 0x30, 0x03, "Cleaning in progess"},
    {0x02, 0x30, 0x07, "Cleaning failure"},
    {0x02, 0x3a, 0x00, "Medium not present"},
    /*KEY=0x03*/
    {0x03, 0x00, 0x00, "Medium Error"},
    /*KEY=0x04*/
    {0x04, 0x00, 0x00, "Hardware Error"},
    /*KEY=0x05*/
    {0x05, 0x00, 0x00, "Illegal Request"},
    /*KEY=0x06*/
    {0x06, 0x00, 0x00, "Unit Attention"},
    /*KEY=0x07*/
    {0x07, 0x00, 0x00, "Data Protect"},
    /*KEY=0x08*/
    {0x08, 0x00, 0x00, "Blank Check"},
    {0x08, 0x00, 0x05, "End of data"},
    {0x08, 0x14, 0x03, "New tape"},
    /*KEY=0x0b*/
    {0x0b, 0x00, 0x00, "Command Aborted"},
    /*KEY=0x0d*/
    {0x0d, 0x00, 0x00, "Volume Overflow"}
};

void _abcdkmt_printf_sense(abcdk_scsi_io_stat *stat)
{
    uint8_t key = 0, asc = 0, ascq = 0;
    const char *msg_p = "Unknown";

    key = abcdk_scsi_sense_key(stat->sense);
    asc = abcdk_scsi_sense_code(stat->sense);
    ascq = abcdk_scsi_sense_qualifier(stat->sense);

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkmt_sense_dict); i++)
    {
        if (abcdkmt_sense_dict[i].key != key)
            continue;

        msg_p = abcdkmt_sense_dict[i].msg;

        if (abcdkmt_sense_dict[i].asc != asc || abcdkmt_sense_dict[i].ascq != ascq)
            continue;

        msg_p = abcdkmt_sense_dict[i].msg;
        break;
    }


    syslog(LOG_INFO, "Sense(KEY=%02X,ASC=%02X,ASCQ=%02X): %s.", key, asc, ascq, msg_p);
}

void _abcdkmt_report_status(abcdk_tree_t *args,int fd)
{
    abcdk_scsi_io_stat stat = {0};
    abcdk_allocator_t *attr_p[6] = {NULL};
    int chk;

    attr_p[0] = abcdk_mt_read_attribute(fd,0,0x0000,3000,&stat);
    if(!attr_p[0] || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    abcdk_endian_b_to_h(attr_p[0]->pptrs[ABCDK_MT_ATTR_VALUE],ABCDK_PTR2U16(attr_p[0]->pptrs[ABCDK_MT_ATTR_LENGTH],0));
    fprintf(stdout,"Remaining_Capacity: %lu\n",ABCDK_PTR2U64(attr_p[0]->pptrs[ABCDK_MT_ATTR_VALUE], 0));
    
    attr_p[1] = abcdk_mt_read_attribute(fd,0,0x0001,3000,&stat);
    if(!attr_p[1] || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    abcdk_endian_b_to_h(attr_p[1]->pptrs[ABCDK_MT_ATTR_VALUE],ABCDK_PTR2U16(attr_p[1]->pptrs[ABCDK_MT_ATTR_LENGTH],0));
    fprintf(stdout,"Maximum_Capacity: %lu\n",ABCDK_PTR2U64(attr_p[1]->pptrs[ABCDK_MT_ATTR_VALUE], 0));
    
    attr_p[2] = abcdk_mt_read_attribute(fd,0,0x0400,3000,&stat);
    if(!attr_p[2] || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    fprintf(stdout,"Manufacturer: %s\n",attr_p[2]->pptrs[ABCDK_MT_ATTR_VALUE]);
    
    attr_p[3] = abcdk_mt_read_attribute(fd,0,0x0401,3000,&stat);
    if(!attr_p[3] || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    fprintf(stdout,"Serial_Number: %s\n",attr_p[3]->pptrs[ABCDK_MT_ATTR_VALUE]);
    

    attr_p[4] = abcdk_mt_read_attribute(fd,0,0x0405,3000,&stat);
    if(!attr_p[4] || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    fprintf(stdout,"Density: %s\n",abcdk_mt_density2string(ABCDK_PTR2U8(attr_p[4]->pptrs[ABCDK_MT_ATTR_VALUE], 0)));
    

    attr_p[5] = abcdk_mt_read_attribute(fd,0,0x0806,3000,&stat);
    if(!attr_p[5] || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    fprintf(stdout,"Barcode: %s\n",attr_p[5]->pptrs[ABCDK_MT_ATTR_VALUE]);
    
   
    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&stat);

final:

    for(size_t i = 0;i<ABCDK_ARRAY_SIZE(attr_p);i++)
    {
        if(!attr_p[i])
            continue;

        abcdk_allocator_unref(&attr_p[i]);
    }
}

void _abcdkmt_read_pos(abcdk_tree_t *args,int fd)
{    
    abcdk_scsi_io_stat stat = {0};
    uint64_t pos_block = -1, pos_file = -1;
    uint32_t pos_part = -1;
    int chk;

    chk = abcdk_mt_read_position(fd,&pos_block,&pos_file,&pos_part,3000,&stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    fprintf(stdout,"Block: %lu\n",pos_block);
    fprintf(stdout,"File: %lu\n",pos_file);
    fprintf(stdout,"Partition: %u\n",pos_part);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&stat);

final:

    return;
}

void _abcdkmt_seek_pos(abcdk_tree_t *args,int fd)
{    
    abcdk_scsi_io_stat stat = {0};
    uint64_t pos_block = INTMAX_MAX;
    uint32_t pos_part = 0;
    int chk;

    pos_block = abcdk_option_get_long(args, "--pos-block", 0, INTMAX_MAX);
    pos_part = abcdk_option_get_int(args, "--pos-part", 0, 0);

    chk = abcdk_mt_locate(fd,1,pos_part,pos_block,1800*1000,&stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

        /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&stat);

final:

    return;
}

void _abcdkmt_work(abcdk_tree_t *args)
{
    abcdk_scsi_io_stat stat = {0};
    uint8_t type = 0;
    char vendor[32] = {0};
    char product[64] = {0};
    char sn[64] = {0};
    int fd = -1;
    const char *dev_p = NULL;
    int filemarks = 0;
    int cmd = 0;
    int chk;

    /*Clear errno.*/
    errno = 0;

    dev_p = abcdk_option_get(args, "--dev", 0, "");
    cmd = abcdk_option_get_int(args, "--cmd", 0, ABCDKMT_STATUS);

    if (access(dev_p, F_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' No such device.", dev_p);
        goto final;
    }

    fd = abcdk_open(dev_p, 1, 1, 0);
    if (fd < 0)
    {
        syslog(LOG_WARNING, "'%s' %s.",dev_p,strerror(errno));
        goto final;
    }

    chk = abcdk_scsi_inquiry_standard(fd, &type, vendor, product, 3000, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    if (type != TYPE_TAPE)
    {
        syslog(LOG_WARNING, "'%s' Not Sequential-Access(tape).", dev_p);
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    chk = abcdk_scsi_inquiry_serial(fd, NULL, sn, 3000, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    if(cmd == ABCDKMT_HWINFO)
    {
        fprintf(stdout,"Vendor: %s\n",vendor);
        fprintf(stdout,"Product: %s\n",product);
        fprintf(stdout,"Serial_Number: %s\n",sn);
    }
    else if (cmd == ABCDKMT_STATUS)
    {
        _abcdkmt_report_status(args, fd);
    }
    else if (cmd == ABCDKMT_REWIND)
    {
        chk = abcdk_mt_rewind(fd,0);
        if (chk != 0)
        {   
            syslog(LOG_WARNING, "'%s' %s.", dev_p, strerror(errno));
            goto final;
        }
    }
    else if (cmd == ABCDKMT_LOAD)
    {
        chk = abcdk_mt_load(fd);
        if (chk != 0)
        {   
            syslog(LOG_WARNING, "'%s' %s.", dev_p, strerror(errno));
            goto final;
        }
    }
    else if (cmd == ABCDKMT_UNLOAD)
    {
        chk = abcdk_mt_unload(fd);
        if (chk != 0)
        {   
            syslog(LOG_WARNING, "'%s' %s.", dev_p, strerror(errno));
            goto final;
        }
    }
    else if (cmd == ABCDKMT_LOCK)
    {
        chk = abcdk_mt_lock(fd);
        if (chk != 0)
        {   
            syslog(LOG_WARNING, "'%s' %s.", dev_p, strerror(errno));
            goto final;
        }
    }
    else if (cmd == ABCDKMT_UNLOCK)
    {
        chk = abcdk_mt_unlock(fd);
        if (chk != 0)
        {   
            syslog(LOG_WARNING, "'%s' %s.", dev_p, strerror(errno));
            goto final;
        }
    }
    else if (cmd == ABCDKMT_READ_POS)
    {
        _abcdkmt_read_pos(args,fd);
    }
    else if (cmd == ABCDKMT_SEEK_POS)
    {
        _abcdkmt_seek_pos(args,fd);
    }
    else if (cmd == ABCDKMT_WRITE_FILEMARK)
    {
        filemarks = abcdk_option_get_int(args, "--filemarks", 0, 1);
        chk = abcdk_mt_writefm(fd,filemarks);
        if (chk != 0)
        {   
            syslog(LOG_WARNING, "'%s' %s.", dev_p, strerror(errno));
            goto final;
        }
    }
    else
    {
        syslog(LOG_WARNING, "Not supported.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&stat);

final:

    abcdk_closep(&fd);
}

int main(int argc, char **argv)
{
    abcdk_tree_t *args;

    args = abcdk_tree_alloc3(1);
    if (!args)
        goto final;

    abcdk_getargs(args, argc, argv, "--");

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(args, "--help"))
    {
        _abcdkmt_print_usage(args, 0);
    }
    else if (abcdk_option_exist(args, "--version"))
    {
        _abcdkmt_print_usage(args, 1);
    }
    else
    {
        _abcdkmt_work(args);
    }

final:

    abcdk_tree_free(&args);

    return errno;
}