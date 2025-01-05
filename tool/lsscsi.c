/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#include "entry.h"

typedef struct _abcdk_lsscsi
{
    int errcode;
    abcdk_option_t *args;

    int fmt;
    const char *outfile;

    abcdk_tree_t *list;

}abcdk_lsscsi_t;

/** 输出格式。*/
enum _abcdk_lsscsi_fmt
{
    /** 文本。*/
    ABCDK_LSSCSI_FMT_TEXT = 1,
#define ABCDK_LSSCSI_FMT_TEXT ABCDK_LSSCSI_FMT_TEXT

    /** XML。*/
    ABCDK_LSSCSI_FMT_XML = 2,
#define ABCDK_LSSCSI_FMT_XML ABCDK_LSSCSI_FMT_XML

    /** JSON。*/
    ABCDK_LSSCSI_FMT_JSON = 3
#define ABCDK_LSSCSI_FMT_JSON ABCDK_LSSCSI_FMT_JSON

};

void _abcdk_lsscsi_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\tSCSI设备枚举器。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到文件(包括路径)。默认：终端\n");

    fprintf(stderr, "\n\t--fmt < FORMAT >\n");
    fprintf(stderr, "\t\t报表格式。默认: %d\n", ABCDK_LSSCSI_FMT_TEXT);

    fprintf(stderr, "\n\t\t%d: TEXT。\n",ABCDK_LSSCSI_FMT_TEXT);
    fprintf(stderr, "\t\t%d: XML。\n",ABCDK_LSSCSI_FMT_XML);
    fprintf(stderr, "\t\t%d: JSON。\n",ABCDK_LSSCSI_FMT_JSON);

    ABCDK_ERRNO_AND_RETURN0(0);
}


int _abcdk_lsscsi_printf_elements_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_lsscsi_t *ctx = (abcdk_lsscsi_t*)opaque;
    abcdk_scsi_info_t *dev_p = NULL;
    
    if(node && node->obj)
        dev_p = (abcdk_scsi_info_t*)node->obj->pptrs[0];

    if (depth == 0)
    {
        if(ctx->fmt == ABCDK_LSSCSI_FMT_XML)
        {
            fprintf(stdout,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            fprintf(stdout,"<devices>\n");
        }
        else if(ctx->fmt == ABCDK_LSSCSI_FMT_JSON)
        {
            fprintf(stdout,"{\n");
            fprintf(stdout,"\"devices\":[\n");
        }
        else if(ctx->fmt == ABCDK_LSSCSI_FMT_TEXT)
        {
            fprintf(stdout, "|%-10s|%-8s|%-10s|%-16s|%-4.4s|%-16s|%-10s\t|%-10s\t|\n",
                    "bus", "type", "vendor", "model", "revision", "serial", "generic", "devname");
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(EINVAL,-1);
        }
    }
    else if (depth == SIZE_MAX)
    {
        if(ctx->fmt == ABCDK_LSSCSI_FMT_XML)
        {
            fprintf(stdout,"</devices>\n");
        }
        else if(ctx->fmt == ABCDK_LSSCSI_FMT_JSON)
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
        if (ctx->fmt == ABCDK_LSSCSI_FMT_XML)
        {
            fprintf(stdout, "\t<device>\n");
            fprintf(stdout, "\t\t<bus>%s</bus>\n",dev_p->bus);
            fprintf(stdout, "\t\t<type name=\"%s\">%u</type>\n",abcdk_scsi_type2string(dev_p->type,0),dev_p->type);
            fprintf(stdout, "\t\t<vendor>%s</vendor>\n",dev_p->vendor);
            fprintf(stdout, "\t\t<model>%s</model>\n",dev_p->model);
            fprintf(stdout, "\t\t<revision>%s</revision>\n",dev_p->revision);
            fprintf(stdout, "\t\t<serial>%s</serial>\n",dev_p->serial);
            fprintf(stdout, "\t\t<devname>%s</devname>\n",dev_p->devname);
            fprintf(stdout, "\t\t<generic>%s</generic>\n",dev_p->generic);
            fprintf(stdout, "\t</device>\n");
        }
        else if(ctx->fmt == ABCDK_LSSCSI_FMT_JSON)
        {
            fprintf(stdout, "\t{\n");
            fprintf(stdout,"\t\t\"bus\":\"%s\",\n",dev_p->bus);
            fprintf(stdout,"\t\t\"type\":{\"num\":\"%u\",\"name\":\"%s\"},\n",dev_p->type,abcdk_scsi_type2string(dev_p->type,0));
            fprintf(stdout,"\t\t\"vendor\":\"%s\",\n",dev_p->vendor);
            fprintf(stdout,"\t\t\"model\":\"%s\",\n",dev_p->model);
            fprintf(stdout,"\t\t\"revison\":\"%s\",\n",dev_p->revision);
            fprintf(stdout,"\t\t\"serial\":\"%s\",\n",dev_p->serial);
            fprintf(stdout,"\t\t\"devname\":\"%s\",\n",dev_p->devname);
            fprintf(stdout,"\t\t\"generic\":\"%s\"\n",dev_p->generic);
            fprintf(stdout, "\t}");
            fprintf(stdout, "%s\n",(abcdk_tree_sibling(node,0)?",":""));
             
        }
        else if(ctx->fmt == ABCDK_LSSCSI_FMT_TEXT)
        {
            fprintf(stdout, "|%-10s|%-8s|%-10s|%-16s|%-4s|%-16s|%-10s\t|%-10s\t|\n",
                    dev_p->bus, abcdk_scsi_type2string(dev_p->type, 0), dev_p->vendor,
                    dev_p->model, dev_p->revision, dev_p->serial, dev_p->generic, dev_p->devname);
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(EINVAL,-1);
        }
    }

    ABCDK_ERRNO_AND_RETURN1(0,1);
}

void _abcdk_lsscsi_printf_elements(abcdk_lsscsi_t *ctx)
{
    abcdk_tree_iterator_t it = {0, ctx, _abcdk_lsscsi_printf_elements_cb};
    abcdk_tree_scan(ctx->list, &it);
}

void _abcdk_lsscsi_work(abcdk_lsscsi_t *ctx)
{
    ctx->outfile = abcdk_option_get(ctx->args, "--output", 0, NULL);
    ctx->fmt = abcdk_option_get_int(ctx->args, "--fmt", 0, ABCDK_LSSCSI_FMT_TEXT);

    abcdk_scsi_watch(&ctx->list,NULL,NULL);
    if(!ctx->list)
        goto final;

    if (ctx->outfile && *ctx->outfile)
    {
        if (abcdk_reopen(STDOUT_FILENO, ctx->outfile, 1, 0, 1) < 0)
        {
            fprintf(stderr, "'%s' %s.\n", ctx->outfile, strerror(errno));
            goto final;
        }
    }

    _abcdk_lsscsi_printf_elements(ctx);

    fflush(stdout);

final:

    abcdk_tree_free(&ctx->list);
}

int abcdk_tool_lsscsi(abcdk_option_t *args)
{
    abcdk_lsscsi_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_lsscsi_print_usage(ctx.args);
    }
    else
    {
        _abcdk_lsscsi_work(&ctx);
    }

    return ctx.errcode;
}