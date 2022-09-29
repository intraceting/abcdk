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
#include "entry.h"


typedef struct _abcdkmtx
{
    int errcode;
    abcdk_tree_t *args;

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

}abcdkmtx_t;


/** 常量。*/
enum _abcdkmtx_constant
{
    /** 打印报表。*/
    ABCDKMTX_STATUS = 1,
#define ABCDKMTX_STATUS ABCDKMTX_STATUS

    /** 移动介质。*/
    ABCDKMTX_MOVE = 2,
#define ABCDKMTX_MOVE ABCDKMTX_MOVE

    /** 文本报表。*/
    ABCDKMTX_STATUS_FMT_TEXT = 1,
#define ABCDKMTX_STATUS_FMT_TEXT ABCDKMTX_STATUS_FMT_TEXT

    /** XML报表。*/
    ABCDKMTX_STATUS_FMT_XML = 2,
#define ABCDKMTX_STATUS_FMT_XML ABCDKMTX_STATUS_FMT_XML

    /** JSON报表。*/
    ABCDKMTX_STATUS_FMT_JSON = 3
#define ABCDKMTX_STATUS_FMT_JSON ABCDKMTX_STATUS_FMT_JSON

};

void _abcdkmtx_print_usage(abcdkmtx_t *ctx)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的机械手(磁带库，光盘库等)工具。\n");

    fprintf(stderr, "\n通用选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--dev < DEVICE >\n");
    fprintf(stderr, "\t\t机械手设备文件(包括路径)。\n");

    fprintf(stderr, "\n\t--cmd < NUMBER >\n");
    fprintf(stderr, "\t\t命令。默认: %d\n", ABCDKMTX_STATUS);

    fprintf(stderr, "\n\t\t%d: 打印报表。\n", ABCDKMTX_STATUS);
    fprintf(stderr, "\t\t%d: 移动介质。\n", ABCDKMTX_MOVE);

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDKMTX_STATUS);

    fprintf(stderr, "\n\t--exclude-barcode\n");
    fprintf(stderr, "\t\t排除条码字段。默认：包括\n");

    fprintf(stderr, "\n\t--exclude-dvcid\n");
    fprintf(stderr, "\t\t排除DVCID字段。默认：包括\n");

    fprintf(stderr, "\n\t--addr-min < ADDRESS >\n");
    fprintf(stderr, "\t\t最小地址(包含)。默认：0\n");

    fprintf(stderr, "\n\t--addr-max < ADDRESS >\n");
    fprintf(stderr, "\t\t最大地址(包含)。默认：65535\n");

    fprintf(stderr, "\n\t--match-type\n");
    fprintf(stderr, "\t\t仅指定类型。默认：全部\n");

    fprintf(stderr, "\n\t\t%d: 机械手。\n", ABCDK_MEDIUMX_ELEMENT_CHANGER);
    fprintf(stderr, "\t\t%d: 存储槽。\n", ABCDK_MEDIUMX_ELEMENT_STORAGE);
    fprintf(stderr, "\t\t%d: 出入槽。\n", ABCDK_MEDIUMX_ELEMENT_IE_PORT);
    fprintf(stderr, "\t\t%d: 驱动器。\n", ABCDK_MEDIUMX_ELEMENT_DXFER);

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到文件(包括路径)。默认：终端\n");

    fprintf(stderr, "\n\t--fmt < FORMAT >\n");
    fprintf(stderr, "\t\t报表格式。默认: %d\n", ABCDKMTX_STATUS_FMT_TEXT);

    fprintf(stderr, "\n\t\t%d: TEXT。\n",ABCDKMTX_STATUS_FMT_TEXT);
    fprintf(stderr, "\t\t%d: XML。\n",ABCDKMTX_STATUS_FMT_XML);
    fprintf(stderr, "\t\t%d: JSON。\n",ABCDKMTX_STATUS_FMT_JSON);

    fprintf(stderr, "\nCMD(%d)选项:\n",ABCDKMTX_MOVE);

    fprintf(stderr, "\n\t--src < ADDRESS >\n");
    fprintf(stderr, "\t\t源地址。\n");

    fprintf(stderr, "\n\t--dst < ADDRESS >\n");
    fprintf(stderr, "\t\t目标地址。\n");
}



void _abcdkmtx_printf_sense(abcdk_scsi_io_stat_t *stat)
{
    uint8_t key = 0, asc = 0, ascq = 0;
    const char *msg_p = NULL;

    key = abcdk_scsi_sense_key(stat->sense);
    asc = abcdk_scsi_sense_code(stat->sense);
    ascq = abcdk_scsi_sense_qualifier(stat->sense);

    msg_p = abcdk_mediumx_sense2string(key, asc, ascq);

    fprintf(stderr, "Sense(KEY=%02X,ASC=%02X,ASCQ=%02X): %s.", key, asc, ascq, (msg_p ? msg_p : "Unknown"));
}

const char *_abcdkmtx_translate_devname(abcdkmtx_t *ctx, uint8_t type, const char *sn)
{
    abcdk_tree_t *node_p = NULL;
    abcdk_scsi_info_t *dev_p = NULL;

    node_p = abcdk_tree_child(ctx->devlist, 1);
    while (node_p)
    {
        dev_p = (abcdk_scsi_info_t *)node_p->alloc->pptrs[0];

        if (dev_p->type == TYPE_TAPE && type == ABCDK_MEDIUMX_ELEMENT_DXFER)
        {
            if (abcdk_strcmp(dev_p->serial, sn, 1) == 0)
                return dev_p->devname;
        }

        if (dev_p->type == TYPE_MEDIUM_CHANGER && type == ABCDK_MEDIUMX_ELEMENT_CHANGER)
        {
            if (abcdk_strcmp(dev_p->serial, sn, 1) == 0)
                return dev_p->generic;
        }

        node_p = abcdk_tree_sibling(node_p, 0);
    }

    return sn;
}

int _abcdkmtx_printf_elements_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdkmtx_t *ctx = (abcdkmtx_t *)opaque;
    const char *sn;
    const char *vendor;
    const char *model;
    uint16_t addr;
    uint8_t type;
    uint8_t full;
    const char *dvcid;
    const char *barcode;

    if (depth == 0)
    {
        sn = (char*)node->alloc->pptrs[0];
        vendor = (char*)node->alloc->pptrs[1];
        model = (char*)node->alloc->pptrs[2];

        if(ctx->fmt == ABCDKMTX_STATUS_FMT_XML)
        {
            fprintf(stdout,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            fprintf(stdout,"<library sn=\"%s\" vendor=\"%s\" model=\"%s\">\n",
                               sn, vendor, model);
            fprintf(stdout,"\t<elements>\n");
        }
        else if(ctx->fmt == ABCDKMTX_STATUS_FMT_JSON)
        {
            fprintf(stdout,"{\n");
            fprintf(stdout,"\t\"sn\":\"%s\",\n",sn);
            fprintf(stdout,"\t\"vendor\":\"%s\",\n",vendor);
            fprintf(stdout,"\t\"model\":\"%s\",\n",model);
            fprintf(stdout,"\t\"elements\":[\n");
        }
        else if(ctx->fmt == ABCDKMTX_STATUS_FMT_TEXT)
        {
            abcdk_tree_fprintf(stdout,depth, node, "%s(%s,%s)\n",
                               sn, vendor, model);
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(ctx->errcode = EINVAL,-1);
        }
    }
    else if (depth == SIZE_MAX)
    {
        if(ctx->fmt == ABCDKMTX_STATUS_FMT_XML)
        {
            fprintf(stdout,"\t</elements>\n");
            fprintf(stdout,"</library>\n");
        }
        else if(ctx->fmt == ABCDKMTX_STATUS_FMT_JSON)
        {
            fprintf(stdout,"\t]\n");
            fprintf(stdout,"}\n");
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(ctx->errcode = EINVAL,-1);
        }
    }
    else
    {
        addr = ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MEDIUMX_ELEMENT_ADDR], 0);
        type = ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MEDIUMX_ELEMENT_TYPE], 0);
        full = ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MEDIUMX_ELEMENT_ISFULL], 0);
        dvcid = _abcdkmtx_translate_devname(ctx,type,(char*)node->alloc->pptrs[ABCDK_MEDIUMX_ELEMENT_DVCID]);
        barcode = (char*)node->alloc->pptrs[ABCDK_MEDIUMX_ELEMENT_BARCODE];

        /*可能仅打印指定类型的元素状态。*/
        if (ctx->match_type != 0 && ctx->match_type != type)
            return 1;

        /*可能仅打印某个范围内的元素状态。*/
        if (addr < ctx->addr_min || addr > ctx->addr_max)
            return 1;

        if (ctx->fmt == ABCDKMTX_STATUS_FMT_XML)
        {
            fprintf(stdout, "\t\t<element address=\"%hu\" type=\"%hhu\" full=\"%hhu\" dvcid=\"%s\" >%s</element>\n",
                    addr,type,full,dvcid,barcode);
        }
        else if(ctx->fmt == ABCDKMTX_STATUS_FMT_JSON)
        {
            fprintf(stdout, "\t\t{\n");
            fprintf(stdout, "\t\t\t\"address\":\"%hu\",\n",addr);
            fprintf(stdout, "\t\t\t\"type\":\"%hhu\",\n",type);
            fprintf(stdout, "\t\t\t\"full\":\"%hhu\",\n",full);
            fprintf(stdout, "\t\t\t\"barcode\":\"%s\"\n",barcode);
            fprintf(stdout, "\t\t\t\"dvcid\":\"%s\",\n",dvcid);
            fprintf(stdout, "\t\t}");
            fprintf(stdout, "%s\n",(abcdk_tree_sibling(node,0)?",":""));
             
        }
        else if(ctx->fmt == ABCDKMTX_STATUS_FMT_TEXT)
        {
            static int print_head = 0;
            if (print_head++ <= 0)
                abcdk_tree_fprintf(stdout, depth, node, "%-6s\t|%-2s\t|%-2s\t|%-7s\t|%-10s\t|\n",
                                   "address", "type", "full", "barcode", "dvcid");

            abcdk_tree_fprintf(stdout, depth, node, "%-6hu\t|%-2hhu\t|%-2hhu\t|%-7s\t|%-10s\t|\n",
                               addr,type,full,barcode,dvcid);
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(ctx->errcode = EINVAL,-1);
        }
    }

    return 1;
}

void _abcdkmtx_printf_elements(abcdkmtx_t *ctx)
{   
    ctx->devlist = abcdk_tree_alloc3(1);
    if (!ctx->devlist)
        return;

    abcdk_scsi_list(ctx->devlist);

    abcdk_tree_iterator_t it = {0, _abcdkmtx_printf_elements_cb, ctx};
    abcdk_tree_scan(ctx->root, &it);

    abcdk_tree_free(&ctx->devlist);
}

int _abcdkmtx_find_changer_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdkmtx_t *ctx = (abcdkmtx_t *)opaque;

    /*已经结束。*/
    if(depth == SIZE_MAX)
        return -1;

    if (depth == 0)
        return 1;

    if (ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MEDIUMX_ELEMENT_TYPE], 0) != ABCDK_MEDIUMX_ELEMENT_CHANGER)
        return 1;

    ctx->changer = ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MEDIUMX_ELEMENT_ADDR], 0);

    return -1;
}

void _abcdkmtx_find_changer(abcdkmtx_t *ctx)
{
    abcdk_tree_iterator_t it = {0, _abcdkmtx_find_changer_cb, ctx};
    abcdk_tree_scan(ctx->root, &it);
}

void _abcdkmtx_move_medium(abcdkmtx_t *ctx)
{
    int chk;

    _abcdkmtx_find_changer(ctx);
    chk = abcdk_mediumx_move_medium(ctx->fd, ctx->changer, ctx->src, ctx->dst, 1800 * 1000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmtx_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdkmtx_work(abcdkmtx_t *ctx)
{
    int chk;

    ctx->fd = -1;
    ctx->dev_p = abcdk_option_get(ctx->args, "--dev", 0, NULL);
    ctx->src = abcdk_option_get_int(ctx->args, "--src", 0, 65536);
    ctx->dst = abcdk_option_get_int(ctx->args, "--dst", 0, 65536);
    ctx->voltag = (abcdk_option_exist(ctx->args, "--exclude-barcode") ? 0 : 1);
    ctx->dvcid = (abcdk_option_exist(ctx->args, "--exclude-dvcid") ? 0 : 1);
    ctx->match_type = abcdk_option_get_int(ctx->args, "--match-type", 0, 0);
    ctx->addr_min = abcdk_option_get_int(ctx->args, "--addr-min", 0, 0);
    ctx->addr_max = abcdk_option_get_int(ctx->args, "--addr-max", 0, 65536);
    ctx->cmd = abcdk_option_get_int(ctx->args, "--cmd", 0, ABCDKMTX_STATUS);
    ctx->fmt = abcdk_option_get_int(ctx->args,"--fmt",0,ABCDKMTX_STATUS_FMT_TEXT);
    ctx->outfile = abcdk_option_get(ctx->args, "--output", 0, NULL);

    if (!ctx->dev_p || !*ctx->dev_p)
    {
        fprintf(stderr, "'--dev DEVICE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->dev_p, F_OK) != 0)
    {
        fprintf(stderr, "'%s' %s。", ctx->dev_p, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    size_t sizes[3] = {100,100,100};
    ctx->root = abcdk_tree_alloc2(sizes,3,0);
    if (!ctx->root)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    ctx->fd = abcdk_open(ctx->dev_p, 0, 0, 0);
    if (ctx->fd < 0)
    {
        fprintf(stderr, "'%s' %s.",ctx->dev_p,strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    chk = abcdk_scsi_inquiry_standard(ctx->fd, &ctx->type, ctx->vendor, ctx->product, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    if (ctx->type != TYPE_MEDIUM_CHANGER)
    {
        fprintf(stderr, "'%s' 不是机械手。", ctx->dev_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,final);
    }

    chk = abcdk_scsi_inquiry_serial(ctx->fd, NULL, ctx->sn, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    snprintf(ctx->root->alloc->pptrs[0], ctx->root->alloc->sizes[0], "%s", ctx->sn);
    snprintf(ctx->root->alloc->pptrs[1], ctx->root->alloc->sizes[1], "%s", ctx->vendor);
    snprintf(ctx->root->alloc->pptrs[2], ctx->root->alloc->sizes[2], "%s", ctx->product);

    chk = abcdk_mediumx_inquiry_element_status(ctx->root, ctx->fd, ctx->voltag,ctx->dvcid,-1, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    if (ctx->cmd == ABCDKMTX_STATUS)
    {
        if (ctx->outfile && *ctx->outfile)
        {
            if (abcdk_reopen(STDOUT_FILENO, ctx->outfile, 1, 0, 1) < 0)
            {
                fprintf(stderr, "'%s' %s。", ctx->outfile, strerror(errno));
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
            }
        }

        _abcdkmtx_printf_elements(ctx);
    }
    else if (ctx->cmd == ABCDKMTX_MOVE)
    {
        _abcdkmtx_move_medium(ctx);
    }
    else
    {
        fprintf(stderr, "尚未支持。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,final);
    }

    fflush(stdout);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmtx_printf_sense(&ctx->stat);

final:

    abcdk_closep(&ctx->fd);
    abcdk_tree_free(&ctx->root);
}

int abcdk_tool_mtx(abcdk_tree_t *args)
{
    abcdkmtx_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkmtx_print_usage(&ctx);
    }
    else
    {
        _abcdkmtx_work(&ctx);
    }
    
    return ctx.errcode;
}