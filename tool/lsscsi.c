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
#include "shell/scsi.h"
#include "entry.h"

typedef struct _abcdklsscsi_ctx
{
    int errcode;

    abcdk_tree_t *args;
}abcdklsscsi_ctx;

void _abcdklsscsi_print_usage(abcdk_tree_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\tSCSI设备枚举器。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdklsscsi_work(abcdklsscsi_ctx *ctx)
{
    abcdk_scsi_info_t devs[10] = {0};
    abcdk_scsi_info_t *dev_p = NULL;
    int count;

    count = abcdk_scsi_list(devs, 10);

    for (int i = 0; i < count; i++)
    {
        dev_p = &devs[i];
        fprintf(stdout, "%-10s,%-20s,%-32s,%-8s,%-16s,%-4s,/dev/%s,/dev/%s\n",
            dev_p->bus,abcdk_scsi_type2string(dev_p->type),dev_p->serial, dev_p->vendor, dev_p->model, dev_p->revision,
            dev_p->devname,dev_p->generic);
    }
}

int abcdk_tool_lsscsi(abcdk_tree_t *args)
{
    abcdklsscsi_ctx ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdklsscsi_print_usage(ctx.args);
    }
    else
    {
        _abcdklsscsi_work(&ctx);
    }

    return ctx.errcode;
}