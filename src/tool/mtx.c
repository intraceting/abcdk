/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"


typedef struct _abcdk_mtx
{
    int errcode;
    abcdk_option_t *args;

    const char *dev_p;
    const char *outfile;
    int src;
    int dst;
    int voltag;
    int dvcid;
    int match_type;
    int addr_min;
    int addr_max;
    int cmd;
    int fmt;

    int fd;
    uint8_t type;
    char vendor[32];
    char product[64];
    char sn[256];
    abcdk_scsi_io_stat_t stat;
    abcdk_tree_t *root;
    uint16_t changer;
    abcdk_tree_t *devlist;

}abcdk_mtx_t;


/** 常量.*/
enum _abcdk_mtx_constant
{
    /** 打印报表.*/
    ABCDK_MTX_STATUS = 1,
#define ABCDK_MTX_STATUS ABCDK_MTX_STATUS

    /** 移动介质.*/
    ABCDK_MTX_MOVE = 2,
#define ABCDK_MTX_MOVE ABCDK_MTX_MOVE

    /** 文本报表.*/
    ABCDK_MTX_STATUS_FMT_TEXT = 1,
#define ABCDK_MTX_STATUS_FMT_TEXT ABCDK_MTX_STATUS_FMT_TEXT

    /** XML报表.*/
    ABCDK_MTX_STATUS_FMT_XML = 2,
#define ABCDK_MTX_STATUS_FMT_XML ABCDK_MTX_STATUS_FMT_XML

    /** JSON报表.*/
    ABCDK_MTX_STATUS_FMT_JSON = 3
#define ABCDK_MTX_STATUS_FMT_JSON ABCDK_MTX_STATUS_FMT_JSON

};

void _abcdk_mtx_print_usage(abcdk_mtx_t *ctx)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的机械手(磁带库, 光盘库等)工具.\n");

    fprintf(stderr, "\n通用选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息.\n");

    fprintf(stderr, "\n\t--dev < DEVICE >\n");
    fprintf(stderr, "\t\t机械手设备文件(包括路径).\n");

    fprintf(stderr, "\n\t--cmd < NUMBER >\n");
    fprintf(stderr, "\t\t命令.默认: %d\n", ABCDK_MTX_STATUS);

    fprintf(stderr, "\n\t\t%d: 打印报表.\n", ABCDK_MTX_STATUS);
    fprintf(stderr, "\t\t%d: 移动介质.\n", ABCDK_MTX_MOVE);

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDK_MTX_STATUS);

    fprintf(stderr, "\n\t--out < FILE >\n");
    fprintf(stderr, "\t\t输出到文件(包括路径).\n");

    fprintf(stderr, "\n\t--fmt < FORMAT >\n");
    fprintf(stderr, "\t\t报表格式.默认: %d\n", ABCDK_MTX_STATUS_FMT_TEXT);

    fprintf(stderr, "\n\t\t%d: TEXT.\n",ABCDK_MTX_STATUS_FMT_TEXT);
    fprintf(stderr, "\t\t%d: XML.\n",ABCDK_MTX_STATUS_FMT_XML);
    fprintf(stderr, "\t\t%d: JSON.\n",ABCDK_MTX_STATUS_FMT_JSON);

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDK_MTX_MOVE);

    fprintf(stderr, "\n\t--src < ADDRESS >\n");
    fprintf(stderr, "\t\t源地址.\n");

    fprintf(stderr, "\n\t--dst < ADDRESS >\n");
    fprintf(stderr, "\t\t目标地址.\n");
}

void _abcdk_mtx_printf_sense(abcdk_scsi_io_stat_t *stat)
{
    abcdk_mediumx_stat_dump(stderr,stat);
}

void _abcdk_mtx_printf_elements(abcdk_mtx_t *ctx)
{   

    abcdk_mediumx_element_status_format(ctx->root,ctx->fmt,stdout);
}

void _abcdk_mtx_find_changer(abcdk_mtx_t *ctx)
{
    ctx->changer = abcdk_mediumx_find_changer_address(ctx->root);
}

void _abcdk_mtx_move_medium(abcdk_mtx_t *ctx)
{
    int chk;

    _abcdk_mtx_find_changer(ctx);
    chk = abcdk_mediumx_move_medium(ctx->fd, ctx->changer, ctx->src, ctx->dst, 1800 * 1000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdk_mtx_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdk_mtx_work(abcdk_mtx_t *ctx)
{
    int chk;

    ctx->fd = -1;
    ctx->dev_p = abcdk_option_get(ctx->args, "--dev", 0, NULL);
    ctx->src = abcdk_option_get_int(ctx->args, "--src", 0, 65536);
    ctx->dst = abcdk_option_get_int(ctx->args, "--dst", 0, 65536);
    ctx->cmd = abcdk_option_get_int(ctx->args, "--cmd", 0, ABCDK_MTX_STATUS);
    ctx->fmt = abcdk_option_get_int(ctx->args,"--fmt",0,ABCDK_MTX_STATUS_FMT_TEXT);
    ctx->outfile = abcdk_option_get(ctx->args, "--out", 0, NULL);

    if (!ctx->dev_p || !*ctx->dev_p)
    {
        fprintf(stderr, "'--dev DEVICE' 不能省略, 且不能为空.\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->dev_p, F_OK) != 0)
    {
        fprintf(stderr, "'%s' %s.\n", ctx->dev_p, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    size_t sizes[3] = {100,100,100};
    ctx->root = abcdk_tree_alloc2(sizes,3,0);
    if (!ctx->root)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    ctx->fd = abcdk_open(ctx->dev_p, 0, 0, 0);
    if (ctx->fd < 0)
    {
        fprintf(stderr, "'%s' %s.\n",ctx->dev_p,strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    chk = abcdk_scsi_inquiry_standard(ctx->fd, &ctx->type, ctx->vendor, ctx->product, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    if (ctx->type != TYPE_MEDIUM_CHANGER)
    {
        fprintf(stderr, "'%s' 不是机械手.\n", ctx->dev_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,final);
    }

    chk = abcdk_scsi_inquiry_serial(ctx->fd, NULL, ctx->sn, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    snprintf(ctx->root->obj->pptrs[0], ctx->root->obj->sizes[0], "%s", ctx->sn);
    snprintf(ctx->root->obj->pptrs[1], ctx->root->obj->sizes[1], "%s", ctx->vendor);
    snprintf(ctx->root->obj->pptrs[2], ctx->root->obj->sizes[2], "%s", ctx->product);

    ctx->voltag = ctx->dvcid = 1;
    
    chk = abcdk_mediumx_inquiry_element_status(ctx->root, ctx->fd, ctx->voltag,ctx->dvcid,-1, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    if (ctx->cmd == ABCDK_MTX_STATUS)
    {
        if (ctx->outfile && *ctx->outfile)
        {
            if (abcdk_reopen(STDOUT_FILENO, ctx->outfile, 1, 0, 1) < 0)
            {
                fprintf(stderr, "'%s' %s.\n", ctx->outfile, strerror(errno));
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
            }
        }

        _abcdk_mtx_printf_elements(ctx);
    }
    else if (ctx->cmd == ABCDK_MTX_MOVE)
    {
        _abcdk_mtx_move_medium(ctx);
    }
    else
    {
        fprintf(stderr, "尚未支持.\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,final);
    }

    fflush(stdout);

    /*No error.*/
    goto final;

print_sense:

    _abcdk_mtx_printf_sense(&ctx->stat);

final:

    abcdk_closep(&ctx->fd);
    abcdk_tree_free(&ctx->root);
}

int abcdk_tool_mtx(abcdk_option_t *args)
{
    abcdk_mtx_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_mtx_print_usage(&ctx);
    }
    else
    {
        _abcdk_mtx_work(&ctx);
    }
    
    return ctx.errcode;
}