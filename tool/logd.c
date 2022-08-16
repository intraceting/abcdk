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
#include "util/thread.h"
#include "util/getargs.h"
#include "util/openssl.h"
#include "shell/file.h"
#include "shell/proc.h"
#include "comm/easy.h"

typedef struct _abcdklogd
{
    int errcode;
    abcdk_tree_t *args;

    const char *ca_file;
    const char *ca_path;
    int ca_check_crl;

    const char *workspace;

    const char *bind;

    abcdk_comm_t *comm;
    abcdk_comm_easy_t *easy;

} abcdklogd_t;

void _abcdklog_print_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);
}

void _abcdklog_work(abcdklogd_t *ctx)
{
    ctx->comm = abcdk_comm_start(0);


    abcdk_comm_stop(&ctx->comm);
}

int abcdk_tool_logd(abcdk_tree_t *args)
{
    abcdklogd_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkbc_print_usage(ctx.args);
    }
    else
    {
        _abcdkbc_work(&ctx);
    }

    return ctx.errcode;
}