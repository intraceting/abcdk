/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"


typedef struct _abcdk_mt
{
    int errcode;
    abcdk_option_t *args;

    const char *dev_p;
    int cmd;

    abcdk_scsi_io_stat_t stat;
    uint8_t type;
    char vendor[32];
    char product[64];
    char sn[256];
    int fd;

}abcdk_mt_t;

/**/
enum _abcdk_mt_constant
{
    /** 倒带。*/
    ABCDK_MT_REWIND = 1,
#define ABCDK_MT_REWIND ABCDK_MT_REWIND

    /** 加载。*/
    ABCDK_MT_LOAD = 2,
#define ABCDK_MT_LOAD ABCDK_MT_LOAD

    /** 卸载。*/
    ABCDK_MT_UNLOAD = 3,
#define ABCDK_MT_UNLOAD ABCDK_MT_UNLOAD

    /** 加锁。*/
    ABCDK_MT_LOCK = 4,
#define ABCDK_MT_LOCK ABCDK_MT_LOCK

    /** 解锁。*/
    ABCDK_MT_UNLOCK = 5,
#define ABCDK_MT_UNLOCK ABCDK_MT_UNLOCK

    /** 读取磁头位置(逻辑)。*/
    ABCDK_MT_TELL_POS = 6,
#define ABCDK_MT_TELL_POS ABCDK_MT_TELL_POS

    /** 移动磁头位置(逻辑)。*/
    ABCDK_MT_SEEK_POS = 7,
#define ABCDK_MT_SEEK_POS ABCDK_MT_SEEK_POS

    /** 写文件标记(filemark)。*/
    ABCDK_MT_WRITE_FILEMARK = 8,
#define ABCDK_MT_WRITE_FILEMARK ABCDK_MT_WRITE_FILEMARK

    /** 读取MAM信息。*/
    ABCDK_MT_READ_MAM = 9,
#define ABCDK_MT_READ_MAM ABCDK_MT_READ_MAM

    /** 写入MAM信息。*/
    ABCDK_MT_WRITE_MAM = 10
#define ABCDK_MT_WRITE_MAM ABCDK_MT_WRITE_MAM

};

void _abcdk_mt_print_usage(abcdk_option_t *args, int only_version)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的磁带驱动器和磁带工具。\n");

    fprintf(stderr, "\n通用选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--dev < DEVICE >\n");
    fprintf(stderr, "\t\t驱动器设备文件(包括路径)。\n");

    fprintf(stderr, "\n\t--cmd < NUMBER >\n");
    fprintf(stderr, "\t\t命令。 默认: %d\n", ABCDK_MT_READ_MAM);

    fprintf(stderr, "\n\t\t%d: 倒带。\n", ABCDK_MT_REWIND);
    fprintf(stderr, "\t\t%d: 加载磁带。\n", ABCDK_MT_LOAD);
    fprintf(stderr, "\t\t%d: 卸载磁带。\n", ABCDK_MT_UNLOAD);
    fprintf(stderr, "\t\t%d: 仓门加锁(禁止磁带被移出驱动器)。\n", ABCDK_MT_LOCK);
    fprintf(stderr, "\t\t%d: 仓门解锁(允许磁带被移出驱动器)。\n", ABCDK_MT_UNLOCK);
    fprintf(stderr, "\t\t%d: 读取磁头位置(逻辑)。\n", ABCDK_MT_TELL_POS);
    fprintf(stderr, "\t\t%d: 移动磁头位置(逻辑)。\n", ABCDK_MT_SEEK_POS);
    fprintf(stderr, "\t\t%d: 写入文件结束标记(filemark)。\n", ABCDK_MT_WRITE_FILEMARK);
    fprintf(stderr, "\t\t%d: 读取MAM信息。\n", ABCDK_MT_READ_MAM);
    fprintf(stderr, "\t\t%d: 写入MAM信息。\n", ABCDK_MT_WRITE_MAM);

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDK_MT_SEEK_POS);

    fprintf(stderr, "\n\t--part < NUMBER >\n");
    fprintf(stderr, "\t\t分区号。 默认: 0\n");

    fprintf(stderr, "\n\t--type < NUMBER >\n");
    fprintf(stderr, "\t\t索引类型。默认: 0\n");

    fprintf(stderr, "\n\t\t0: 逻辑块。\n");
    fprintf(stderr, "\t\t1: 逻辑文件。\n");

    fprintf(stderr, "\n\t--pos < NUMBER >\n");
    fprintf(stderr, "\t\t索引位置。默认: 末尾\n");

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDK_MT_WRITE_FILEMARK);

    fprintf(stderr, "\n\t--count < NUMBER >\n");
    fprintf(stderr, "\t\t数量。 默认: 1\n");

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDK_MT_READ_MAM);

    fprintf(stderr, "\n\t--part < NUMBER >\n");
    fprintf(stderr, "\t\t分区号。 默认: 0\n");
    
    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDK_MT_WRITE_MAM);

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
    fprintf(stderr, "\t\t内容(TEXT,ASCII)。\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdk_mt_printf_sense(abcdk_scsi_io_stat_t *stat)
{
    abcdk_tape_stat_dump(stderr,stat);
}

void _abcdk_mt_operate(abcdk_mt_t *ctx)
{
    int chk;

    if(ctx->cmd == ABCDK_MT_REWIND)
        chk = abcdk_tape_operate(ctx->fd, MTREW, 0, &ctx->stat);
    else if(ctx->cmd == ABCDK_MT_LOAD)
        chk = abcdk_tape_load(ctx->fd,1,60*1000,&ctx->stat);
    else if(ctx->cmd == ABCDK_MT_UNLOAD)
        chk = abcdk_tape_load(ctx->fd,2,180*1000,&ctx->stat);
    else if(ctx->cmd == ABCDK_MT_LOCK)
        chk = abcdk_tape_operate(ctx->fd, MTLOCK, 0, &ctx->stat);
    else if(ctx->cmd == ABCDK_MT_UNLOCK)
        chk = abcdk_tape_operate(ctx->fd, MTUNLOCK, 0, &ctx->stat);

    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdk_mt_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdk_mt_write_filemark(abcdk_mt_t *ctx)
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

    _abcdk_mt_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdk_mt_tell_pos(abcdk_mt_t *ctx)
{    
    abcdk_scsi_io_stat_t stat = {0};
    uint64_t block = -1, file = -1;
    uint32_t part = -1;
    int chk;

    chk = abcdk_tape_tell(ctx->fd,&block,&file,&part,3000,&ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);
    
    fprintf(stdout,"|%-10s\t|%-10s\t|%-10s\t|\n","block","file","part");
    fprintf(stdout,"|%-10lu\t|%-10lu\t|%-10u\t|\n",block,file,part);

    /*No error.*/
    goto final;

print_sense:

    _abcdk_mt_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdk_mt_seek_pos(abcdk_mt_t *ctx)
{    
    int part;
    uint8_t type;
    uint64_t pos;
    int chk;

    part = abcdk_option_get_int(ctx->args, "--part", 0,0);
    type = abcdk_option_get_int(ctx->args, "--type", 0,0);
    pos = abcdk_option_get_llong(ctx->args, "--pos", 0, INTMAX_MAX);
    
    chk = abcdk_tape_seek(ctx->fd, 1, type, part, pos, 1800 * 1000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdk_mt_printf_sense(&ctx->stat);

final:

    return;
}


int _abcdk_mt_printf_mam_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_mt_t *ctx = (abcdk_mt_t *)opaque;
    uint16_t id,len;
    uint8_t rd ,fmt;
    uint8_t *val;
    uint64_t val_int = 0;
    uint8_t val_buf[400] = {0};
    char *rd_str[] = {"RW","RO"};
    char *fmt_str[] = {"BINARY","ASCII","TEXT","Reserved"};

    if (depth == 0)
    {
        fprintf(stdout,"|%-4s|%-40s\t|%-2s\t|%-5s\t|%-5s\t|%-40s\t|\n","id","name","ro/rw","format","length","value");
    }
    else if (depth == SIZE_MAX)
    {
        return -1;
    }
    else
    {
        id = ABCDK_PTR2U16(node->obj->pptrs[ABCDK_TAPE_ATTR_ID], 0);
        rd = ABCDK_PTR2U8(node->obj->pptrs[ABCDK_TAPE_ATTR_READONLY], 0);
        fmt = ABCDK_PTR2U8(node->obj->pptrs[ABCDK_TAPE_ATTR_FORMAT], 0);
        len = ABCDK_PTR2U16(node->obj->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0);
        val = node->obj->pptrs[ABCDK_TAPE_ATTR_VALUE];

        if (len <= 0)
            return 1;

        fprintf(stdout,"|%04X|%-40s\t|%-2s\t|%-5s\t|%-5hu\t",id,abcdk_tape_attr2string(id),rd_str[rd],fmt_str[fmt],len);

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
                sprintf(val_buf,"%llu(%#llx),%s",val_int,val_int,abcdk_tape_density2string(val_int));
                fprintf(stdout, "|%-40s\t|", val_buf);
            }
            else if(id == 0x0408)
            {
                sprintf(val_buf,"%llu(%#llx),%s",val_int,val_int,abcdk_tape_type2string(val_int));
                fprintf(stdout, "|%-40s\t|", val_buf);
            }
            else if (len <= 8)
            {
                sprintf(val_buf,"%llu(%#llx)",val_int,val_int);
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

abcdk_tree_t *_abcdk_mt_read_mam_one(abcdk_mt_t *ctx, uint8_t part, uint16_t id)
{
    abcdk_tree_t *node = NULL;

    node = abcdk_tree_alloc3(1);
    if(!node)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = ENOMEM, final_error);

    node->obj = abcdk_tape_read_attribute(ctx->fd, part, id, 3000, &ctx->stat);
    if (!node->obj || ctx->stat.status != GOOD)
    {
        fprintf(stderr,"Read MAM(id(%04x),part(%hhu)) failed. \n",id,part);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, print_sense);
    }

    return node;

print_sense:
    
    _abcdk_mt_printf_sense(&ctx->stat);

final_error:

    abcdk_tree_free(&node);

    return NULL;
}

void _abcdk_mt_read_mam(abcdk_mt_t *ctx)
{
    abcdk_object_t *attr_p = NULL;
    abcdk_tree_t *root = NULL, *node = NULL;
    abcdk_tree_iterator_t it = {0, ctx, _abcdk_mt_printf_mam_cb};
    int part;
    int chk;

    part = abcdk_option_get_int(ctx->args, "--part", 0,0);

    root = abcdk_tree_alloc3(1);
    if(!root)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = ENOMEM, final);


    for (size_t i = 0; i < 65536; i++)
    {
        if (!abcdk_tape_attr2string(i))
            continue;

        node = _abcdk_mt_read_mam_one(ctx,part,i);
        if(node)
        {
            abcdk_tree_insert2(root, node, 0);
        }
        else 
        {
            /*如果磁带没准备好，直接跳出。*/
            if (abcdk_scsi_sense_key(ctx->stat.sense) == 0x02 &&
                abcdk_scsi_sense_code(ctx->stat.sense) == 0x3A &&
                abcdk_scsi_sense_qualifier(ctx->stat.sense) == 0x00)
                goto final2;
        }
    }

final:

    /*打印。*/
    abcdk_tree_scan(root, &it);

final2:

    abcdk_tree_free(&root);
}

void _abcdk_mt_write_mam(abcdk_mt_t *ctx)
{   
    int id = 0xffff;
    const char *value = NULL;
    int val_len = -1;
    abcdk_object_t *attr_p = NULL;
    int chk;

    id = abcdk_option_get_int(ctx->args, "--id", 0, 0xFFFF);
    value = abcdk_option_get(ctx->args, "--value", 0, "");
    val_len = strlen(value);

    if (!abcdk_tape_attr2string(id))
    {
        fprintf(stderr, "'--id < NUMBER >' 不能省略，且不能为空，同时必须在有效范围内。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (val_len <= 0)
    {
        fprintf(stderr, "没有输入ID的值，MAM中ID的值将被清空。\n");
    }

    attr_p = abcdk_tape_read_attribute(ctx->fd,0,id,3000,&ctx->stat);
    if(!attr_p || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);
    
    memset(attr_p->pptrs[ABCDK_TAPE_ATTR_VALUE],0,ABCDK_PTR2U16(attr_p->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0));

    if (val_len > ABCDK_PTR2U16(attr_p->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0))
    {
        val_len = ABCDK_PTR2U16(attr_p->pptrs[ABCDK_TAPE_ATTR_LENGTH], 0);
        fprintf(stderr, "ID的值将被截断为%d字节。\n",val_len);
    }

    memcpy(attr_p->pptrs[ABCDK_TAPE_ATTR_VALUE],value,val_len);

    chk = abcdk_tape_write_attribute(ctx->fd,0,attr_p,3000,&ctx->stat);
    if(chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdk_mt_printf_sense(&ctx->stat);

final:

    abcdk_object_unref(&attr_p);

    return;
}

static struct _abcdk_mt_methods
{
    int cmd;
    void (*method)(abcdk_mt_t *ctx);
} abcdk_mt_methods[] = {
    {ABCDK_MT_REWIND,_abcdk_mt_operate},
    {ABCDK_MT_LOAD,_abcdk_mt_operate},
    {ABCDK_MT_UNLOAD,_abcdk_mt_operate},
    {ABCDK_MT_LOCK,_abcdk_mt_operate},
    {ABCDK_MT_UNLOCK,_abcdk_mt_operate},
    {ABCDK_MT_WRITE_FILEMARK,_abcdk_mt_write_filemark},
    {ABCDK_MT_TELL_POS, _abcdk_mt_tell_pos},
    {ABCDK_MT_SEEK_POS, _abcdk_mt_seek_pos},
    {ABCDK_MT_READ_MAM, _abcdk_mt_read_mam},
    {ABCDK_MT_WRITE_MAM, _abcdk_mt_write_mam},
};

void _abcdk_mt_work(abcdk_mt_t *ctx)
{
    void (*_method)(abcdk_mt_t *ctx) = NULL;
    int chk;

    ctx->fd = -1;
    ctx->dev_p = abcdk_option_get(ctx->args, "--dev", 0, "");
    ctx->cmd = abcdk_option_get_int(ctx->args, "--cmd", 0, ABCDK_MT_READ_MAM);

    if (!ctx->dev_p || !*ctx->dev_p)
    {
        fprintf(stderr, "'--dev DEVICE' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (access(ctx->dev_p, F_OK) != 0)
    {
        fprintf(stderr, "'%s' %s。\n", ctx->dev_p, strerror(errno));
        goto final;
    }

    ctx->fd = abcdk_open(ctx->dev_p, 1, 1, 0);
    if (ctx->fd < 0)
    {
        fprintf(stderr, "'%s' %s.\n",ctx->dev_p,strerror(errno));
        goto final;
    }

    chk = abcdk_scsi_inquiry_standard(ctx->fd, &ctx->type, ctx->vendor, ctx->product, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    if (ctx->type != TYPE_TAPE)
    {
        fprintf(stderr, "'%s' 不是磁带驱动器。\n", ctx->dev_p);
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    chk = abcdk_scsi_inquiry_serial(ctx->fd, NULL, ctx->sn, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    fprintf(stderr,"Driver: %s(%s,%s)\n",ctx->sn,ctx->vendor,ctx->product);

    /*加载磁带前不需要执行测试。*/
    if(ctx->cmd != ABCDK_MT_LOAD)
    {
        chk = abcdk_scsi_test(ctx->fd,1000,&ctx->stat);
        if (chk != 0 || ctx->stat.status != GOOD)
            ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);
    }

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_mt_methods); i++)
    {
        if (abcdk_mt_methods[i].cmd != ctx->cmd)
            continue;

        _method = abcdk_mt_methods[i].method;
        break;
    }

    if (!_method)
    {
        fprintf(stderr, "CMD(%d)尚未支持。\n", ctx->cmd);
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    _method(ctx);

    /*No error.*/
    goto final;

print_sense:

    _abcdk_mt_printf_sense(&ctx->stat);

final:

    abcdk_closep(&ctx->fd);
}

int abcdk_tool_mt(abcdk_option_t *args)
{
    abcdk_mt_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_mt_print_usage(ctx.args, 0);
    }
    else
    {
        _abcdk_mt_work(&ctx);
    }

    return ctx.errcode;
}