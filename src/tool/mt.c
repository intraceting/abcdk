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
#include <locale.h>
#include "util/general.h"
#include "util/getargs.h"
#include "util/scsi.h"
#include "util/tape.h"
#include "entry.h"


typedef struct _abcdkmtx_ctx
{
    int errcode;
    abcdk_tree_t *args;

    const char *dev_p;
    int cmd;

    abcdk_scsi_io_stat stat;
    uint8_t type;
    char vendor[32];
    char product[64];
    char sn[256];
    int fd;

}abcdkmtx_ctx;

/**/
enum _abcdkmt_constant
{
    /** 倒带。*/
    ABCDKMT_REWIND = 1,
#define ABCDKMT_REWIND ABCDKMT_REWIND

    /** 加载。*/
    ABCDKMT_LOAD = 2,
#define ABCDKMT_LOAD ABCDKMT_LOAD

    /** 卸载。*/
    ABCDKMT_UNLOAD = 3,
#define ABCDKMT_UNLOAD ABCDKMT_UNLOAD

    /** 加锁。*/
    ABCDKMT_LOCK = 4,
#define ABCDKMT_LOCK ABCDKMT_LOCK

    /** 解锁。*/
    ABCDKMT_UNLOCK = 5,
#define ABCDKMT_UNLOCK ABCDKMT_UNLOCK

    /** 读取磁头位置(逻辑)。*/
    ABCDKMT_TELL_POS = 6,
#define ABCDKMT_TELL_POS ABCDKMT_TELL_POS

    /** 移动磁头位置(逻辑)。*/
    ABCDKMT_SEEK_POS = 7,
#define ABCDKMT_SEEK_POS ABCDKMT_SEEK_POS

    /** 写文件标记(filemark)。*/
    ABCDKMT_WRITE_FILEMARK = 8,
#define ABCDKMT_WRITE_FILEMARK ABCDKMT_WRITE_FILEMARK

    /** 读取MAM信息。*/
    ABCDKMT_READ_MAM = 9,
#define ABCDKMT_READ_MAM ABCDKMT_READ_MAM

    /** 写入MAM信息。*/
    ABCDKMT_WRITE_MAM = 10,
#define ABCDKMT_WRITE_MAM ABCDKMT_WRITE_MAM

};

void _abcdkmt_print_usage(abcdk_tree_t *args, int only_version)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的磁带驱动器和磁带工具。\n");

    fprintf(stderr, "\n通用选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--dev < DEVICE >\n");
    fprintf(stderr, "\t\t驱动器设备文件(包括路径)。\n");

    fprintf(stderr, "\n\t--cmd < NUMBER >\n");
    fprintf(stderr, "\t\t命令。 默认: %d\n", ABCDKMT_READ_MAM);

    fprintf(stderr, "\t\t%d: 倒带。\n", ABCDKMT_REWIND);
    fprintf(stderr, "\t\t%d: 加载磁带。\n", ABCDKMT_LOAD);
    fprintf(stderr, "\t\t%d: 卸载磁带。\n", ABCDKMT_UNLOAD);
    fprintf(stderr, "\t\t%d: 仓门加锁(禁止磁带被移出驱动器)。\n", ABCDKMT_LOCK);
    fprintf(stderr, "\t\t%d: 仓门解锁(允许磁带被移出驱动器)。\n", ABCDKMT_UNLOCK);
    fprintf(stderr, "\t\t%d: 读取磁头位置(逻辑)。\n", ABCDKMT_TELL_POS);
    fprintf(stderr, "\t\t%d: 移动磁头位置(逻辑)。\n", ABCDKMT_SEEK_POS);
    fprintf(stderr, "\t\t%d: 写入文件标记(filemark)。\n", ABCDKMT_WRITE_FILEMARK);
    fprintf(stderr, "\t\t%d: 读取MAM信息。\n", ABCDKMT_READ_MAM);
    fprintf(stderr, "\t\t%d: 写入MAM信息。\n", ABCDKMT_WRITE_MAM);

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDKMT_SEEK_POS);

    fprintf(stderr, "\n\t--block < NUMBER >\n");
    fprintf(stderr, "\t\t块索引。默认: 末尾\n");

    fprintf(stderr, "\n\t--partition < NUMBER >\n");
    fprintf(stderr, "\t\t分区号。 默认: 0\n");

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDKMT_WRITE_FILEMARK);

    fprintf(stderr, "\n\t--count < NUMBER >\n");
    fprintf(stderr, "\t\t数量。 默认: 1\n");

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDKMT_READ_MAM);

    fprintf(stderr, "\n\t--id < NUMBER >\n");
    fprintf(stderr, "\t\t编号。默认：全部\n");

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDKMT_WRITE_MAM);

    fprintf(stderr, "\n\t--id < NUMBER >\n");
    fprintf(stderr, "\t\t编号。\n");

    fprintf(stderr, "\n\t--value < VALUE >\n");
    fprintf(stderr, "\t\t内容(TEXT,ASCII,BINARY)。\n");

    ABCDK_ERRNO_AND_RETURN0(0);
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

static struct _abcdkmt_mam_dict
{
    uint16_t id;
    const char *name;
}abcdkmt_mam_dict[]={
    /*MAM Device type attributes*/
    {0x0000,"REMAINING CAPACITY IN PARTITION"},
    {0x0001,"MAXIMUM CAPACITY IN PARTITION"},
    {0x0002,"TAPEALERT FLAGS"},
    {0x0003,"LOAD COUNT"},
    {0x0004,"MAM SPACE REMAINING"},
    {0x0005,"ASSIGNING ORGANIZATION"},
    {0x0006,"FORMATTED DENSITY CODE"},
    {0x0007,"INITIALIZATION COUNT"},
    {0x0009,"VOLUME CHANGE REFERENCE"},
    {0x020A,"DEVICE VENDOR/SERIAL NUMBER AT LAST LOAD"},
    {0x020B,"DEVICE VENDOR/SERIAL NUMBER AT LOAD-1"},
    {0x020C,"DEVICE VENDOR/SERIAL NUMBER AT LOAD-2"},
    {0x020D,"DEVICE VENDOR/SERIAL NUMBER AT LOAD-3"},
    {0x0220,"TOTAL MBYTES WRITTEN IN MEDIUM LIFE"},
    {0x0221,"TOTAL MBYTES READ IN MEDIUM LIFE"},
    {0x0222,"TOTAL MBYTES WRITTEN IN CURRENT/LAST LOAD"},
    {0x0223,"TOTAL MBYTES READ IN CURRENT/LAST LOAD"},
    /*MAM Medium type attributes*/
    {0x0400,"MEDIUM MANUFACTURER"},
    {0x0401,"MEDIUM SERIAL NUMBER"},
    {0x0402,"MEDIUM LENGTH"},
    {0x0403,"MEDIUM WIDTH"},
    {0x0404,"ASSIGNING ORGANIZATION"},
    {0x0405,"MEDIUM DENSITY CODE"},
    {0x0406,"MEDIUM MANUFACTURE DATE"},
    {0x0407,"MAM CAPACITY"},
    {0x0408,"MEDIUM TYPE"},
    {0x0409,"MEDIUM TYPE INFORMATION"},
    /*MAM Host type attributes*/
    {0x0800,"APPLICATION VENDOR"},
    {0x0801,"APPLICATION NAME"},
    {0x0802,"APPLICATION VERSION"},
    {0x0803,"USER MEDIUM TEXT LABEL"},
    {0x0804,"DATE AND TIME LAST WRITTEN"},
    {0x0805,"TEXT LOCALIZATION IDENTIFIER"},
    {0x0806,"BARCODE(条码)"},
    {0x0807,"OWNING HOST TEXTUAL NAME"},
    {0x0808,"MEDIA POOL"},
    {0x080B,"APPLICATION FORMAT VERSION"},
    {0x080C,"VOLUME COHERENCY INFORMATION"},
    {0x0820,"MEDIUM GLOBALLY UNIQUE IDENTIFIER"},
    {0x0821,"MEDIA POOL GLOBALLY UNIQUE IDENTIFIER"}
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

void _abcdkmt_operate(abcdkmtx_ctx *ctx)
{
    int chk;

    if(ctx->cmd == ABCDKMT_REWIND)
        chk = abcdk_tape_operate(ctx->fd, MTREW, 0, &ctx->stat);
    else if(ctx->cmd == ABCDKMT_LOAD)
        chk = abcdk_tape_operate(ctx->fd, MTLOAD, 0, &ctx->stat);
    else if(ctx->cmd == ABCDKMT_UNLOAD)
        chk = abcdk_tape_operate(ctx->fd, MTUNLOAD, 0, &ctx->stat);
    else if(ctx->cmd == ABCDKMT_LOCK)
        chk = abcdk_tape_operate(ctx->fd, MTLOCK, 0, &ctx->stat);
    else if(ctx->cmd == ABCDKMT_UNLOCK)
        chk = abcdk_tape_operate(ctx->fd, MTUNLOCK, 0, &ctx->stat);

    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdkmt_write_filemark(abcdkmtx_ctx *ctx)
{
    int count;
    int chk;

    count = abcdk_option_get_int(ctx->args, "--count", 0, 1);

    chk = abcdk_tape_operate(ctx->fd, MTWEOF, count, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdkmt_tell_pos(abcdkmtx_ctx *ctx)
{    
    abcdk_scsi_io_stat stat = {0};
    uint64_t block = -1, file = -1;
    uint32_t part = -1;
    int chk;

    chk = abcdk_tape_tell(ctx->fd,&block,&file,&part,3000,&ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);
    
    fprintf(stdout,"|%-10s\t|%-10s\t|%-10s\t|\n","BLock","Logical File","Partition");
    fprintf(stdout,"|%-10lu\t|%-10lu\t|%-10u\t|\n",block,file,part);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdkmt_seek_pos(abcdkmtx_ctx *ctx)
{    
    uint64_t block;
    uint32_t part;
    int chk;

    block = abcdk_option_get_llong(ctx->args, "--block", 0, INTMAX_MAX);
    part = abcdk_option_get_int(ctx->args, "--partition", 0, 0);

    chk = abcdk_tape_seek(ctx->fd, 1, part, block, 1800 * 1000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&ctx->stat);

final:

    return;
}


int _abcdkmt_printf_mam_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdkmtx_ctx *ctx = (abcdkmtx_ctx *)opaque;
    uint16_t id;
    uint8_t rd;
    uint8_t fmt;
    uint16_t len;
    uint8_t *val;

    if (depth == 0)
    {
        fprintf(stdout,"|%-4s |%-1s |%-1s |%-5% |%-40s|\n","id","ro","fmt","length","value");
    }
    else if (depth == SIZE_MAX)
    {
        
    }
    else
    {
        id = ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_TAPE_ATTR_ID], 0);
        rd = ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_TAPE_ATTR_READONLY], 0);
        fmt = ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_TAPE_ATTR_FORMAT], 0);
        len = ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0);
        val = node->alloc->pptrs[ABCDK_TAPE_ATTR_VALUE];

        fprintf(stdout,"|%-04x |%-1hhu |%-1hhu |%-5hu | \n", id,rd,fmt,len);
        
    }
}

abcdk_tree_t *_abcdkmt_read_mam_one(abcdkmtx_ctx *ctx,int id)
{
    abcdk_tree_t *node = NULL;

    node = abcdk_tree_alloc3(1);
    if(!node)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = ENOMEM, final_error);

    node->alloc = abcdk_tape_read_attribute(ctx->fd, 0, id, 3000, &ctx->stat);
    if (!node->alloc || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, print_sense);

    return node;

print_sense:

    _abcdkmt_printf_sense(&ctx->stat);

final_error:

    abcdk_tree_free(&node);
}

void _abcdkmt_read_mam(abcdkmtx_ctx *ctx)
{
    abcdk_allocator_t *attr_p = NULL;
    abcdk_tree_t *root = NULL, *node = NULL;
    abcdk_tree_iterator_t it = {0, _abcdkmt_printf_mam_cb, ctx};
    int id;
    int chk;

    root = abcdk_tree_alloc3(1);
    if(!root)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = ENOMEM, final);

    id = abcdk_option_get_int(ctx->args, "--id", 0, -1);
    if (id >= 0)
    {
        node = _abcdkmt_read_mam_one(ctx, id);
        if (!node)
            goto final;

        abcdk_tree_insert2(root,node,0);
    }
    else 
    {
        for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkmt_mam_dict); i++)
        {
            node = _abcdkmt_read_mam_one(ctx, abcdkmt_mam_dict[i].id);
            if (!node)
            {
                /*如果磁带没准备好，直接跳出。*/
                if (abcdk_scsi_sense_key(ctx->stat.sense) == 0x02 &&
                    abcdk_scsi_sense_code(ctx->stat.sense) == 0x3A &&
                    abcdk_scsi_sense_qualifier(ctx->stat.sense) == 0x00)
                    break;
                else
                    continue;
            }

            abcdk_tree_insert2(root,node,0);
        }
    }

final:

    /*打印。*/
    abcdk_tree_scan(root, &it);

    abcdk_tree_free(&root);
}

void _abcdkmt_write_barcode(abcdk_tree_t *args,int fd)
{   
    abcdk_scsi_io_stat stat = {0};
    const char *barcode_p = NULL;
    abcdk_allocator_t *attr_p = NULL;
    int chk;

    barcode_p = abcdk_option_get(args, "--barcode", 0, "");

    if (!barcode_p || !*barcode_p)
    {
        syslog(LOG_ERR, "'--barcode STRING' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (strlen(barcode_p) > 32)
    {
        syslog(LOG_ERR, "条码长度不能超过32个字符。");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    attr_p = abcdk_tape_read_attribute(fd,0,0x0806,3000,&stat);
    if(!attr_p || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    
    memcpy(attr_p->pptrs[ABCDK_TAPE_ATTR_VALUE],barcode_p,ABCDK_PTR2U16(attr_p->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0));

    chk = abcdk_tape_write_attribute(fd,0,attr_p,3000,&stat);
    if(chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&stat);

final:

    abcdk_allocator_unref(&attr_p);

    return;
}

static struct _abcdkmt_methods
{
    int cmd;
    void (*method)(abcdkmtx_ctx *ctx);
} abcdkmt_methods[] = {
    {ABCDKMT_REWIND,_abcdkmt_operate},
    {ABCDKMT_LOAD,_abcdkmt_operate},
    {ABCDKMT_UNLOAD,_abcdkmt_operate},
    {ABCDKMT_LOCK,_abcdkmt_operate},
    {ABCDKMT_UNLOCK,_abcdkmt_operate},
    {ABCDKMT_WRITE_FILEMARK,_abcdkmt_write_filemark},
    {ABCDKMT_TELL_POS, _abcdkmt_tell_pos},
    {ABCDKMT_SEEK_POS, _abcdkmt_seek_pos},
    {ABCDKMT_READ_MAM,_abcdkmt_read_mam}
};

void _abcdkmt_work(abcdkmtx_ctx *ctx)
{
    void (*_method)(abcdkmtx_ctx *ctx) = NULL;
    int chk;

    ctx->fd = -1;
    ctx->dev_p = abcdk_option_get(ctx->args, "--dev", 0, "");
    ctx->cmd = abcdk_option_get_int(ctx->args, "--cmd", 0, ABCDKMT_READ_MAM);

    if (!ctx->dev_p || !*ctx->dev_p)
    {
        syslog(LOG_ERR, "'--dev DEVICE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (access(ctx->dev_p, F_OK) != 0)
    {
        syslog(LOG_ERR, "'%s' %s。", ctx->dev_p, strerror(errno));
        goto final;
    }

    ctx->fd = abcdk_open(ctx->dev_p, 1, 1, 0);
    if (ctx->fd < 0)
    {
        syslog(LOG_ERR, "'%s' %s.",ctx->dev_p,strerror(errno));
        goto final;
    }

    chk = abcdk_scsi_inquiry_standard(ctx->fd, &ctx->type, ctx->vendor, ctx->product, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    if (ctx->type != TYPE_TAPE)
    {
        syslog(LOG_ERR, "'%s' 不是磁带驱动器。", ctx->dev_p);
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    chk = abcdk_scsi_inquiry_serial(ctx->fd, NULL, ctx->sn, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    syslog(LOG_INFO,"Driver: %s(%s,%s)",ctx->sn,ctx->vendor,ctx->product);

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkmt_methods); i++)
    {
        if (abcdkmt_methods[i].cmd != ctx->cmd)
            continue;

        _method = abcdkmt_methods[i].method;
        break;
    }

    if (!_method)
    {
        syslog(LOG_ERR, "CMD(%d)尚未支持。", ctx->cmd);
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    _method(ctx);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&ctx->stat);

final:

    abcdk_closep(&ctx->fd);
}

int abcdk_tool_mt(abcdk_tree_t *args)
{
    abcdkmtx_ctx ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkmt_print_usage(ctx.args, 0);
    }
    else
    {
        _abcdkmt_work(&ctx);
    }

    return ctx.errcode;
}