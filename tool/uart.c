/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "entry.h"

typedef struct _abcdkuart
{
    int errcode;
    abcdk_option_t *args;

    int fd;
    int baudrate;
    int bits;
    int parity;
    int stop;

}abcdkuart_t;

void _abcdkuart_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的串口工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");


}

void _abcdkuart_wrok(abcdkuart_t *ctx)
{
    const char *dev_p = abcdk_option_get(ctx->args,"--dev",0,"");

    ctx->fd = -1;



    abcdk_closep(&ctx->fd);
}

int abcdk_tool_uart(abcdk_option_t *args)
{
    abcdkuart_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkuart_print_usage(ctx.args);
    }
    else
    {
        _abcdkuart_wrok(&ctx);
    }


    return ctx.errcode;
}