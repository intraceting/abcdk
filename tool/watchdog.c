/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "entry.h"

typedef struct _abcdkwatchdog
{
    int errcode;
    abcdk_option_t *args;

    

}abcdkwatchdog_t;

void _abcdkwatchdog_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的看门狗。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

}

void _abcdkwatchdog_work_process(abcdkwatchdog_t *ctx,int idx)
{
    const char *conf_p = abcdk_option_get(ctx->args,"--conf",idx,"");
}

void _abcdkwatchdog_wrok(abcdkwatchdog_t *ctx)
{
   
}

int abcdk_tool_watchdog(abcdk_option_t *args)
{
    abcdkwatchdog_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkwatchdog_print_usage(ctx.args);
    }
    else
    {
        _abcdkwatchdog_wrok(&ctx);
    }


    return ctx.errcode;
}