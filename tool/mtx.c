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
#include "entry.h"

/* */
#define _(string) gettext(string)

/** 命令。*/
enum _abcdkmtx_cmd
{
    /** 枚举磁带库所有元素状态。*/
    ABCDKMTX_STATUS = 1,
#define ABCDKMTX_STATUS ABCDKMTX_STATUS

    /** 移动磁带。*/
    ABCDKMTX_MOVE = 2
#define ABCDKMTX_MOVE ABCDKMTX_MOVE

};

/** 元素状态格式。*/
enum _abcdkmtx_status_fmt
{
    /** 文本。*/
    ABCDKMTX_STATUS_FMT_TEXT = 1,
#define ABCDKMTX_STATUS_FMT_TEXT ABCDKMTX_STATUS_FMT_TEXT

    /** XML。*/
    ABCDKMTX_STATUS_FMT_XML = 2,
#define ABCDKMTX_STATUS_FMT_XML ABCDKMTX_STATUS_FMT_XML

    /** JSON。*/
    ABCDKMTX_STATUS_FMT_JSON = 3
#define ABCDKMTX_STATUS_FMT_JSON ABCDKMTX_STATUS_FMT_JSON

};

void _abcdkmtx_print_usage(abcdk_tree_t *args, int only_version)
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

    fprintf(stderr, "\n\t--cmd < NUMBER >\n");
    fprintf(stderr, "\t\t命令。默认: %d\n", ABCDKMTX_STATUS);

    fprintf(stderr, "\n\t\t%d: 打印报表。\n", ABCDKMTX_STATUS);
    fprintf(stderr, "\t\t%d: 移动介质。\n", ABCDKMTX_MOVE);

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到指定的文件(包括路径)。默认：终端\n");

    fprintf(stderr, "\n\t--fmt < FORMAT >\n");
    fprintf(stderr, "\t\t指定报表格式。默认: %d\n", ABCDKMTX_STATUS_FMT_TEXT);

    fprintf(stderr, "\n\t\t%d: TEXT。\n",ABCDKMTX_STATUS_FMT_TEXT);
    fprintf(stderr, "\t\t%d: XML。\n",ABCDKMTX_STATUS_FMT_XML);
    fprintf(stderr, "\t\t%d: JSON。\n",ABCDKMTX_STATUS_FMT_JSON);

    ABCDK_ERRNO_AND_RETURN0(0);
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

int _abcdkmtx_printf_elements_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    long fmt = (long)opaque;

    if (depth == 0)
    {
        if(fmt == ABCDKMTX_STATUS_FMT_XML)
        {
            fprintf(stdout,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            fprintf(stdout,"<library sn=\"%s\" vendor=\"%s\" model=\"%s\">\n",
                               node->alloc->pptrs[0], node->alloc->pptrs[1], node->alloc->pptrs[2]);
            fprintf(stdout,"\t<elements>\n");
        }
        else if(fmt == ABCDKMTX_STATUS_FMT_JSON)
        {
            fprintf(stdout,"{\n");
            fprintf(stdout,"\t\"sn\":\"%s\",\n",node->alloc->pptrs[0]);
            fprintf(stdout,"\t\"vendor\":\"%s\",\n",node->alloc->pptrs[1]);
            fprintf(stdout,"\t\"model\":\"%s\",\n",node->alloc->pptrs[2]);
            fprintf(stdout,"\t\"elements\":[\n");
        }
        else if(fmt == ABCDKMTX_STATUS_FMT_TEXT)
        {
            abcdk_tree_fprintf(stdout,depth, node, "%s(%s,%s)\n",
                               node->alloc->pptrs[0], node->alloc->pptrs[1], node->alloc->pptrs[2]);
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(EINVAL,-1);
        }
    }
    else if (depth == SIZE_MAX)
    {
        if(fmt == ABCDKMTX_STATUS_FMT_XML)
        {
            fprintf(stdout,"\t</elements>\n");
            fprintf(stdout,"</library>\n");
        }
        else if(fmt == ABCDKMTX_STATUS_FMT_JSON)
        {
            fprintf(stdout,"\t]\n");
            fprintf(stdout,"}\n");
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(EINVAL,-1);
        }
    }
    else
    {
        if (fmt == ABCDKMTX_STATUS_FMT_XML)
        {
            fprintf(stdout, "\t\t<element addr=\"%hu\" type=\"%hhu\" isfull=\"%hhu\" dvcid=\"%s\" >%s</element>\n",
                    ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0),
                    ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0),
                    ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ISFULL], 0),
                    node->alloc->pptrs[ABCDK_MTX_ELEMENT_DVCID],
                    node->alloc->pptrs[ABCDK_MTX_ELEMENT_BARCODE]);
        }
        else if(fmt == ABCDKMTX_STATUS_FMT_JSON)
        {
            fprintf(stdout, "\t\t{\n");
            fprintf(stdout, "\t\t\t\"addr\":\"%hu\",\n",ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0));
            fprintf(stdout, "\t\t\t\"type\":\"%hhu\",\n",ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0));
            fprintf(stdout, "\t\t\t\"isfull\":\"%hhu\",\n",ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ISFULL], 0));
            fprintf(stdout, "\t\t\t\"barcode\":\"%s\",\n",node->alloc->pptrs[ABCDK_MTX_ELEMENT_BARCODE]);
            fprintf(stdout, "\t\t\t\"dvcid\":\"%s\"\n",node->alloc->pptrs[ABCDK_MTX_ELEMENT_DVCID]);
            fprintf(stdout, "\t\t}");
            fprintf(stdout, "%s\n",(abcdk_tree_sibling(node,0)?",":""));
             
        }
        else if(fmt == ABCDKMTX_STATUS_FMT_TEXT)
        {
            abcdk_tree_fprintf(stdout, depth, node, "%-6hu\t|%-2hhu\t|%-2hhu\t|%-10s\t|%-10s\t|\n",
                               ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0),
                               ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0),
                               ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ISFULL], 0),
                               node->alloc->pptrs[ABCDK_MTX_ELEMENT_BARCODE],
                               node->alloc->pptrs[ABCDK_MTX_ELEMENT_DVCID]);
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(EINVAL,-1);
        }
    }

    ABCDK_ERRNO_AND_RETURN1(0,1);
}

void _abcdkmtx_printf_elements(abcdk_tree_t *args,abcdk_tree_t *root)
{
    long fmt = abcdk_option_get_long(args,"--fmt",0,ABCDKMTX_STATUS_FMT_TEXT);

    abcdk_tree_iterator_t it = {0, _abcdkmtx_printf_elements_cb, (void*)fmt};
    abcdk_tree_scan(root, &it);
}

int _abcdkmtx_find_changer_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    uint16_t *t_p = (uint16_t *)opaque;

    /*已经结束。*/
    if(depth == SIZE_MAX)
        ABCDK_ERRNO_AND_RETURN1(0,-1);

    if (depth == 0)
        ABCDK_ERRNO_AND_RETURN1(0,1);

    if (ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0) != ABCDK_MXT_ELEMENT_CHANGER)
        ABCDK_ERRNO_AND_RETURN1(0,1);

    *t_p = ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0);

    ABCDK_ERRNO_AND_RETURN1(0,-1);
}

uint16_t _abcdkmtx_find_changer(abcdk_tree_t *root)
{
    uint16_t t = 0;

    abcdk_tree_iterator_t it = {0, _abcdkmtx_find_changer_cb, &t};
    abcdk_tree_scan(root, &it);

    /*Clear errno.*/
    errno = 0;

    return t;
}

void _abcdkmtx_move_medium(abcdk_tree_t *args, int fd, abcdk_tree_t *root)
{
    abcdk_scsi_io_stat stat = {0};
    int t = 0, s = 65536, d = 65536;
    int chk;

    s = abcdk_option_get_int(args, "--src", 0, 65536);
    d = abcdk_option_get_int(args, "--dst", 0, 65536);

    /*Clear errno.*/
    errno = 0;

    t = _abcdkmtx_find_changer(root);
    chk = abcdk_mtx_move_medium(fd, t, s, d, 1800 * 1000, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EINVAL,print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmtx_printf_sense(&stat);

final:

    return;
}

void _abcdkmtx_work(abcdk_tree_t *args)
{
    abcdk_scsi_io_stat stat = {0};
    abcdk_tree_t *root = NULL;
    uint8_t type = 0;
    char vendor[32] = {0};
    char product[64] = {0};
    char sn[64] = {0};
    int fd = -1;
    const char *dev_p = NULL;
    int voltag = 1;
    int dvcid = 1;
    int cmd = 0;
    const char *outfile = NULL;
    int chk;

    dev_p = abcdk_option_get(args, "--dev", 0, NULL);
    voltag = (abcdk_option_exist(args, "--exclude-barcode") ? 0 : 1);
    dvcid = (abcdk_option_exist(args, "--exclude-dvcid") ? 0 : 1);
    cmd = abcdk_option_get_int(args, "--cmd", 0, ABCDKMTX_STATUS);
    outfile = abcdk_option_get(args, "--output", 0, NULL);

    /*Clear errno.*/
    errno = 0;

    if (!dev_p || !*dev_p)
    {
        syslog(LOG_ERR, "'--dev FILE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (access(dev_p, F_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s。", dev_p, strerror(errno));
        goto final;
    }

    size_t sizes[3] = {100,100,100};
    root = abcdk_tree_alloc2(sizes,3,0);
    if (!root)
        goto final;

    fd = abcdk_open(dev_p, 0, 0, 0);
    if (fd < 0)
    {
        syslog(LOG_WARNING, "'%s' %s.",dev_p,strerror(errno));
        goto final;
    }

    chk = abcdk_scsi_inquiry_standard(fd, &type, vendor, product, 3000, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    if (type != TYPE_MEDIUM_CHANGER)
    {
        syslog(LOG_WARNING, "'%s' 不是机械手.", dev_p);
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    chk = abcdk_scsi_inquiry_serial(fd, NULL, sn, 3000, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    snprintf(root->alloc->pptrs[0], root->alloc->sizes[0], "%s", sn);
    snprintf(root->alloc->pptrs[1], root->alloc->sizes[1], "%s", vendor);
    snprintf(root->alloc->pptrs[2], root->alloc->sizes[2], "%s", product);

    chk = abcdk_mtx_inquiry_element_status(root, fd, voltag,dvcid,-1, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    if (cmd == ABCDKMTX_STATUS)
    {
        if (outfile && *outfile)
        {
            if (abcdk_reopen(STDOUT_FILENO, outfile, 1, 0, 1) < 0)
            {
                syslog(LOG_WARNING, "'%s' %s.", outfile, strerror(errno));
                goto final;
            }
        }

        _abcdkmtx_printf_elements(args,root);
    }
    else if (cmd == ABCDKMTX_MOVE)
    {
        _abcdkmtx_move_medium(args, fd, root);
    }
    else
    {
        syslog(LOG_WARNING, "尚未支持。");
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    fflush(stdout);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmtx_printf_sense(&stat);

final:

    abcdk_closep(&fd);
    abcdk_tree_free(&root);
}

int abcdk_tool_mtx(abcdk_tree_t *args)
{
    if (abcdk_option_exist(args, "--help"))
    {
        _abcdkmtx_print_usage(args, 0);
    }
    else
    {
        _abcdkmtx_work(args);
    }
    
    return errno;
}