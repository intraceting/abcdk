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

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDKMT_WRITE_MAM);

    fprintf(stderr, "\n\t--id < NUMBER >\n");
    fprintf(stderr, "\t\t编号。\n");

    fprintf(stderr, "\n");
    for (int i = 0,j = 0; i < 65536; i++)
    {
        const char *str = abcdk_tape_attr2string(i);
        if (!str)
            continue;

        fprintf(stderr,"\t\t%04X: %-40s",i,str);
        if ((++j) % 2 == 0)
            fprintf(stderr, "\n");
    }

    fprintf(stderr, "\n\t--value < VALUE >\n");
    fprintf(stderr, "\t\t内容(TEXT,ASCII,BINARY)。\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdkmt_printf_sense(abcdk_scsi_io_stat *stat)
{
    uint8_t key = 0, asc = 0, ascq = 0;
    const char *msg_p = NULL;

    key = abcdk_scsi_sense_key(stat->sense);
    asc = abcdk_scsi_sense_code(stat->sense);
    ascq = abcdk_scsi_sense_qualifier(stat->sense);

    msg_p = abcdk_tape_sense2string(key, asc, ascq);

    syslog(LOG_INFO, "Sense(KEY=%02X,ASC=%02X,ASCQ=%02X): %s.", key, asc, ascq, (msg_p ? msg_p : "Unknown"));
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
    
    fprintf(stdout,"|%-10s\t|%-10s\t|%-10s\t|\n","block","file","partition");
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
    uint16_t id,len;
    uint8_t rd ,fmt;
    uint8_t *val;
    uint64_t val_int = 0;
    uint8_t val_buf[400] = {0};
    char *rd_str[] = {"RW","RO"};
    char *fmt_str[] = {"BINARY","ASCII","TEXT","Reserved"};

    if (depth == 0)
    {
        fprintf(stdout,"|%-40s\t|%-2s\t|%-5s\t|%-5s\t|%-40s\t|\n","name","ro/rw","format","length","value");
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

        if (len <= 0)
            return 1;

        fprintf(stdout,"|%-40s\t|%-2s\t|%-5s\t|%-5hu\t", abcdk_tape_attr2string(id),rd_str[rd],fmt_str[fmt],len);

        if (fmt == 0x00)
        {
            if(len == 1)
                val_int = ABCDK_PTR2U8(val,0);
            else if(len == 2)
                val_int = abcdk_endian_b_to_h16(ABCDK_PTR2U16(val,0));
            else if(len == 4)
                val_int = abcdk_endian_b_to_h32(ABCDK_PTR2U32(val,0));
            else if(len == 8)
                val_int = abcdk_endian_b_to_h64(ABCDK_PTR2U64(val,0));
            else
                abcdk_bin2hex(val_buf,val,len,0);

            if(id == 0x0006 || id == 0x0405)
            {
                sprintf(val_buf,"%lu(%#lx),%s",val_int,val_int,abcdk_tape_density2string(val_int));
                fprintf(stdout, "|%-40s\t|", val_buf);
            }
            else if(id == 0x0408)
            {
                sprintf(val_buf,"%lu(%#lx),%s",val_int,val_int,abcdk_tape_type2string(val_int));
                fprintf(stdout, "|%-40s\t|", val_buf);
            }
            else if (len <= 8)
            {
                sprintf(val_buf,"%lu(%#lx)",val_int,val_int);
                fprintf(stdout, "|%-40s\t|", val_buf);
            }
            else if (len <= 40)
                fprintf(stdout, "|%-40s\t|", val_buf);
            else
                fprintf(stdout, "|%-37.37s...\t|", val_buf);
        }
        else if (fmt == 0x01)
        {
            if (strlen(val) <= 40)
                fprintf(stdout, "|%-40s\t|",val);
            else
                fprintf(stdout, "|%-37.37s\t|",val);
        }
        else if (fmt == 0x02)
        {
            if (strlen(val) <= 40)
                fprintf(stdout, "|%-40s\t|",val);
            else
                fprintf(stdout, "|%-37.37s\t|",val);
        }

        fprintf(stdout,"\n");
        
    }

    return 1;
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

    return NULL;
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

    for (size_t i = 0; i < 65536; i++)
    {
        if (!abcdk_tape_attr2string(i))
            continue;

        node = _abcdkmt_read_mam_one(ctx, i);
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

        abcdk_tree_insert2(root, node, 0);
    }

final:

    /*打印。*/
    abcdk_tree_scan(root, &it);

    abcdk_tree_free(&root);
}

void _abcdkmt_write_mam(abcdkmtx_ctx *ctx)
{   
    int id = 0xffff;
    const char *value = NULL;
    int val_len = -1;
    abcdk_allocator_t *attr_p = NULL;
    int chk;

    id = abcdk_option_get_int(ctx->args,"--id",0,0xFFFF);
    value = abcdk_option_get(ctx->args, "--value", 0, "");
    val_len = strlen(value);

    if (!abcdk_tape_attr2string(id))
    {
        syslog(LOG_ERR, "'--id < NUMBER >' 不能省略，且不能为空，同时必须在有效范围内。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (val_len <= 0)
    {
        syslog(LOG_WARNING, "没有输入ID值，MAM中ID的值将被清空。");
    }

    attr_p = abcdk_tape_read_attribute(ctx->fd,0,id,3000,&ctx->stat);
    if(!attr_p || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);
    
    memset(attr_p->pptrs[ABCDK_TAPE_ATTR_VALUE],0,ABCDK_PTR2U16(attr_p->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0));

    if (val_len > ABCDK_PTR2U16(attr_p->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0))
    {
        val_len = ABCDK_PTR2U16(attr_p->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0);
        syslog(LOG_WARNING, "ID的值将被截断为%d字节。",val_len);
    }

    memcpy(attr_p->pptrs[ABCDK_TAPE_ATTR_VALUE],value,val_len);

    chk = abcdk_tape_write_attribute(ctx->fd,0,attr_p,3000,&ctx->stat);
    if(chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmt_printf_sense(&ctx->stat);

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
    {ABCDKMT_READ_MAM, _abcdkmt_read_mam}
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