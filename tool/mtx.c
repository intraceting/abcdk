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
#include "util/mtx.h"
#include "shell/scsi.h"
#include "entry.h"


typedef struct _abcdkmtx_ctx
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
    abcdk_scsi_io_stat stat;
    abcdk_tree_t *root;
    uint16_t changer;
    abcdk_tree_t *devlist;

}abcdkmtx_ctx;


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

void _abcdkmtx_print_usage(abcdkmtx_ctx *ctx)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的机械手(磁带库，光盘库等)工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--dev < FILE >\n");
    fprintf(stderr, "\t\t机械手设备文件(包括路径)。\n");

    fprintf(stderr, "\n\t--src < ADDRESS >\n");
    fprintf(stderr, "\t\t源地址。\n");

    fprintf(stderr, "\n\t--dst < ADDRESS >\n");
    fprintf(stderr, "\t\t目标地址。\n");

    fprintf(stderr, "\n\t--exclude-barcode\n");
    fprintf(stderr, "\t\t排除条码字段。默认：包括\n");

    fprintf(stderr, "\n\t--exclude-dvcid\n");
    fprintf(stderr, "\t\t排除DVCID字段。默认：包括\n");

    fprintf(stderr, "\n\t--addr-min < ADDRESS >\n");
    fprintf(stderr, "\t\t最小地址(包含)。默认：0\n");

    fprintf(stderr, "\n\t--addr-max < ADDRESS >\n");
    fprintf(stderr, "\t\t最大地址(包含)。默认：65536\n");

    fprintf(stderr, "\n\t--match-type\n");
    fprintf(stderr, "\t\t仅指定类型。默认：全部\n");

    fprintf(stderr, "\n\t\t%d: 机械手。\n", ABCDK_MXT_ELEMENT_CHANGER);
    fprintf(stderr, "\n\t\t%d: 存储槽位。\n", ABCDK_MXT_ELEMENT_STORAGE);
    fprintf(stderr, "\n\t\t%d: IE槽位。\n", ABCDK_MXT_ELEMENT_IE_PORT);
    fprintf(stderr, "\n\t\t%d: 驱动器。\n", ABCDK_MXT_ELEMENT_DXFER);

    fprintf(stderr, "\n\t--cmd < NUMBER >\n");
    fprintf(stderr, "\t\t命令。默认: %d\n", ABCDKMTX_STATUS);

    fprintf(stderr, "\n\t\t%d: 打印报表。\n", ABCDKMTX_STATUS);
    fprintf(stderr, "\t\t%d: 移动介质。\n", ABCDKMTX_MOVE);

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到文件(包括路径)。默认：终端\n");

    fprintf(stderr, "\n\t--fmt < FORMAT >\n");
    fprintf(stderr, "\t\t报表格式。默认: %d\n", ABCDKMTX_STATUS_FMT_TEXT);

    fprintf(stderr, "\n\t\t%d: TEXT。\n",ABCDKMTX_STATUS_FMT_TEXT);
    fprintf(stderr, "\t\t%d: XML。\n",ABCDKMTX_STATUS_FMT_XML);
    fprintf(stderr, "\t\t%d: JSON。\n",ABCDKMTX_STATUS_FMT_JSON);
}

static struct _abcdkmtx_sense_dict
{   
    uint8_t key;
    uint8_t asc;
    uint8_t ascq;
    const char *msg;
}abcdkmtx_sense_dict[] = {
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
    {0x05, 0x24, 0x00, "无效的地址或地址超出范围"},
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

void _abcdkmtx_printf_sense(abcdk_scsi_io_stat *stat)
{
    uint8_t key = 0, asc = 0, ascq = 0;
    const char *msg_p = "Unknown";

    key = abcdk_scsi_sense_key(stat->sense);
    asc = abcdk_scsi_sense_code(stat->sense);
    ascq = abcdk_scsi_sense_qualifier(stat->sense);

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkmtx_sense_dict); i++)
    {
        if (abcdkmtx_sense_dict[i].key != key)
            continue;

        msg_p = abcdkmtx_sense_dict[i].msg;

        if (abcdkmtx_sense_dict[i].asc != asc || abcdkmtx_sense_dict[i].ascq != ascq)
            continue;

        msg_p = abcdkmtx_sense_dict[i].msg;
        break;
    }

    syslog(LOG_INFO, "Sense(KEY=%02X,ASC=%02X,ASCQ=%02X): %s.", key, asc, ascq, msg_p);
}

const char *_abcdkmtx_translate_devname(abcdkmtx_ctx *ctx, uint8_t type, const char *sn)
{
    abcdk_tree_t *node_p = NULL;
    abcdk_scsi_info_t *dev_p = NULL;

    node_p = abcdk_tree_child(ctx->devlist, 1);
    while (node_p)
    {
        dev_p = (abcdk_scsi_info_t *)node_p->alloc->pptrs[0];

        if (dev_p->type == TYPE_TAPE && type == ABCDK_MXT_ELEMENT_DXFER)
        {
            if (abcdk_strcmp(dev_p->serial, sn, 1) == 0)
                return dev_p->devname;
        }

        if (dev_p->type == TYPE_MEDIUM_CHANGER && type == ABCDK_MXT_ELEMENT_CHANGER)
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
    abcdkmtx_ctx *ctx = (abcdkmtx_ctx *)opaque;
    const char *sn;
    const char *vendor;
    const char *model;
    uint16_t addr;
    uint8_t type;
    uint8_t isfull;
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
        addr = ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0);
        type = ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0);
        isfull = ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ISFULL], 0);
        dvcid = _abcdkmtx_translate_devname(ctx,type,(char*)node->alloc->pptrs[ABCDK_MTX_ELEMENT_DVCID]);
        barcode = (char*)node->alloc->pptrs[ABCDK_MTX_ELEMENT_BARCODE];

        /*可能仅打印指定类型的元素状态。*/
        if (ctx->match_type != 0 && ctx->match_type != type)
            return 1;

        /*可能仅打印某个范围内的元素状态。*/
        if (addr < ctx->addr_min || addr > ctx->addr_max)
            return 1;

        if (ctx->fmt == ABCDKMTX_STATUS_FMT_XML)
        {
            fprintf(stdout, "\t\t<element addr=\"%hu\" type=\"%hhu\" isfull=\"%hhu\" dvcid=\"%s\" >%s</element>\n",
                    addr,type,isfull,dvcid,barcode);
        }
        else if(ctx->fmt == ABCDKMTX_STATUS_FMT_JSON)
        {
            fprintf(stdout, "\t\t{\n");
            fprintf(stdout, "\t\t\t\"addr\":\"%hu\",\n",addr);
            fprintf(stdout, "\t\t\t\"type\":\"%hhu\",\n",type);
            fprintf(stdout, "\t\t\t\"isfull\":\"%hhu\",\n",isfull);
            fprintf(stdout, "\t\t\t\"dvcid\":\"%s\",\n",dvcid);
            fprintf(stdout, "\t\t\t\"barcode\":\"%s\"\n",barcode);
            fprintf(stdout, "\t\t}");
            fprintf(stdout, "%s\n",(abcdk_tree_sibling(node,0)?",":""));
             
        }
        else if(ctx->fmt == ABCDKMTX_STATUS_FMT_TEXT)
        {
            abcdk_tree_fprintf(stdout, depth, node, "%-6hu\t|%-2hhu\t|%-2hhu\t|%-10s\t|%-10s\t|\n",
                               addr,type,isfull,dvcid,barcode);
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(ctx->errcode = EINVAL,-1);
        }
    }

    return 1;
}

void _abcdkmtx_printf_elements(abcdkmtx_ctx *ctx)
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
    abcdkmtx_ctx *ctx = (abcdkmtx_ctx *)opaque;

    /*已经结束。*/
    if(depth == SIZE_MAX)
        return -1;

    if (depth == 0)
        return 1;

    if (ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0) != ABCDK_MXT_ELEMENT_CHANGER)
        return 1;

    ctx->changer = ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0);

    return -1;
}

void _abcdkmtx_find_changer(abcdkmtx_ctx *ctx)
{
    abcdk_tree_iterator_t it = {0, _abcdkmtx_find_changer_cb, ctx};
    abcdk_tree_scan(ctx->root, &it);
}

void _abcdkmtx_move_medium(abcdkmtx_ctx *ctx)
{
    int chk;

    _abcdkmtx_find_changer(ctx);
    chk = abcdk_mtx_move_medium(ctx->fd, ctx->changer, ctx->src, ctx->dst, 1800 * 1000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmtx_printf_sense(&ctx->stat);

final:

    return;
}

void _abcdkmtx_work(abcdkmtx_ctx *ctx)
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
        syslog(LOG_ERR, "'--dev FILE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->dev_p, F_OK) != 0)
    {
        syslog(LOG_ERR, "'%s' %s。", ctx->dev_p, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    size_t sizes[3] = {100,100,100};
    ctx->root = abcdk_tree_alloc2(sizes,3,0);
    if (!ctx->root)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    ctx->fd = abcdk_open(ctx->dev_p, 0, 0, 0);
    if (ctx->fd < 0)
    {
        syslog(LOG_ERR, "'%s' %s.",ctx->dev_p,strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    chk = abcdk_scsi_inquiry_standard(ctx->fd, &ctx->type, ctx->vendor, ctx->product, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    if (ctx->type != TYPE_MEDIUM_CHANGER)
    {
        syslog(LOG_ERR, "'%s' 不是机械手.", ctx->dev_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,final);
    }

    chk = abcdk_scsi_inquiry_serial(ctx->fd, NULL, ctx->sn, 3000, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    snprintf(ctx->root->alloc->pptrs[0], ctx->root->alloc->sizes[0], "%s", ctx->sn);
    snprintf(ctx->root->alloc->pptrs[1], ctx->root->alloc->sizes[1], "%s", ctx->vendor);
    snprintf(ctx->root->alloc->pptrs[2], ctx->root->alloc->sizes[2], "%s", ctx->product);

    chk = abcdk_mtx_inquiry_element_status(ctx->root, ctx->fd, ctx->voltag,ctx->dvcid,-1, &ctx->stat);
    if (chk != 0 || ctx->stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM,print_sense);

    if (ctx->cmd == ABCDKMTX_STATUS)
    {
        if (ctx->outfile && *ctx->outfile)
        {
            if (abcdk_reopen(STDOUT_FILENO, ctx->outfile, 1, 0, 1) < 0)
            {
                syslog(LOG_ERR, "'%s' %s.", ctx->outfile, strerror(errno));
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
        syslog(LOG_INFO, "尚未支持。");
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
    abcdkmtx_ctx ctx = {0};

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