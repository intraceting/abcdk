/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "entry.h"

typedef struct _abcdk_lsmmc
{
    int errcode;
    abcdk_option_t *args;

    int fmt;
    const char *outfile;

    abcdk_tree_t *list;

}abcdk_lsmmc_t;

/** 输出格式。*/
enum _abcdk_lsmmc_fmt
{
    /** 文本。*/
    ABCDK_LSMMC_FMT_TEXT = 1,
#define ABCDK_LSMMC_FMT_TEXT ABCDK_LSMMC_FMT_TEXT

    /** XML。*/
    ABCDK_LSMMC_FMT_XML = 2,
#define ABCDK_LSMMC_FMT_XML ABCDK_LSMMC_FMT_XML

    /** JSON。*/
    ABCDK_LSMMC_FMT_JSON = 3
#define ABCDK_LSMMC_FMT_JSON ABCDK_LSMMC_FMT_JSON

};

void _abcdk_lsmmc_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\tMMC设备枚举器。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到文件(包括路径)。默认：终端\n");

    fprintf(stderr, "\n\t--fmt < FORMAT >\n");
    fprintf(stderr, "\t\t报表格式。默认: %d\n", ABCDK_LSMMC_FMT_TEXT);

    fprintf(stderr, "\n\t\t%d: TEXT。\n",ABCDK_LSMMC_FMT_TEXT);
    fprintf(stderr, "\t\t%d: XML。\n",ABCDK_LSMMC_FMT_XML);
    fprintf(stderr, "\t\t%d: JSON。\n",ABCDK_LSMMC_FMT_JSON);

    ABCDK_ERRNO_AND_RETURN0(0);
}


int _abcdk_lsmmc_printf_elements_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_lsmmc_t *ctx = (abcdk_lsmmc_t*)opaque;
    abcdk_mmc_info_t *dev_p = NULL;
    
    if(node && node->obj)
        dev_p = (abcdk_mmc_info_t*)node->obj->pptrs[0];

    if (depth == 0)
    {
        if(ctx->fmt == ABCDK_LSMMC_FMT_XML)
        {
            fprintf(stdout,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            fprintf(stdout,"<devices>\n");
        }
        else if(ctx->fmt == ABCDK_LSMMC_FMT_JSON)
        {
            fprintf(stdout,"{\n");
            fprintf(stdout,"\"devices\":[\n");
        }
        else if(ctx->fmt == ABCDK_LSMMC_FMT_TEXT)
        {
            fprintf(stdout, "|%-10s|%-8s|%-16s|%-32s|%-10s\t|\n",
                    "bus", "type", "name", "cid", "devname");
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(EINVAL,-1);
        }
    }
    else if (depth == SIZE_MAX)
    {
        if(ctx->fmt == ABCDK_LSMMC_FMT_XML)
        {
            fprintf(stdout,"</devices>\n");
        }
        else if(ctx->fmt == ABCDK_LSMMC_FMT_JSON)
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
        if (ctx->fmt == ABCDK_LSMMC_FMT_XML)
        {
            fprintf(stdout, "\t<device>\n");
            fprintf(stdout, "\t\t<bus>%s</bus>\n",dev_p->bus);
            fprintf(stdout, "\t\t<type>%s</type>\n",dev_p->type);
            fprintf(stdout, "\t\t<name>%s</name>\n",dev_p->name);
            fprintf(stdout, "\t\t<cid>%s</cid>\n",dev_p->cid);
            fprintf(stdout, "\t\t<devname>%s</devname>\n",dev_p->devname);
            fprintf(stdout, "\t</device>\n");
        }
        else if(ctx->fmt == ABCDK_LSMMC_FMT_JSON)
        {
            fprintf(stdout, "\t{\n");
            fprintf(stdout,"\t\t\"bus\":\"%s\",\n",dev_p->bus);
            fprintf(stdout,"\t\t\"type\":\"%s\",\n",dev_p->type);
            fprintf(stdout,"\t\t\"name\":\"%s\",\n",dev_p->name);
            fprintf(stdout,"\t\t\"cid\":\"%s\",\n",dev_p->cid);
            fprintf(stdout,"\t\t\"devname\":\"%s\"\n",dev_p->devname);
            fprintf(stdout, "\t}");
            fprintf(stdout, "%s\n",(abcdk_tree_sibling(node,0)?",":""));
             
        }
        else if(ctx->fmt == ABCDK_LSMMC_FMT_TEXT)
        {
            fprintf(stdout, "|%-10s|%-8s|%-16s|%-32s|%-10s\t|\n",
                    dev_p->bus, dev_p->type, dev_p->name, dev_p->cid, dev_p->devname);
        }
        else
        {
            ABCDK_ERRNO_AND_RETURN1(EINVAL,-1);
        }
    }

    ABCDK_ERRNO_AND_RETURN1(0,1);
}

void _abcdk_lsmmc_printf_elements(abcdk_lsmmc_t *ctx)
{
    abcdk_tree_iterator_t it = {0, ctx, _abcdk_lsmmc_printf_elements_cb};
    abcdk_tree_scan(ctx->list, &it);
}

void _abcdk_lsmmc_work(abcdk_lsmmc_t *ctx)
{
    ctx->outfile = abcdk_option_get(ctx->args, "--output", 0, NULL);
    ctx->fmt = abcdk_option_get_int(ctx->args, "--fmt", 0, ABCDK_LSMMC_FMT_TEXT);

    abcdk_mmc_watch(&ctx->list,NULL,NULL);
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

    _abcdk_lsmmc_printf_elements(ctx);

    fflush(stdout);

final:

    abcdk_tree_free(&ctx->list);
}

int abcdk_tool_lsmmc(abcdk_option_t *args)
{
    abcdk_lsmmc_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_lsmmc_print_usage(ctx.args);
    }
    else
    {
        _abcdk_lsmmc_work(&ctx);
    }

    return ctx.errcode;
}