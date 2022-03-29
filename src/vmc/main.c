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
#include "context.h"
#include "service.h"


int main(int argc, char **argv)
{
    abcdk_vmtx_t ctx = {0};

    /*中文；UTF-8。*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    /*随机数种子。*/
    srand(time(NULL));

    /*申请参数存储空间。*/
    ctx.args = abcdk_tree_alloc3(1);
    if (!ctx.args)
        ABCDK_ERRNO_AND_GOTO1(ctx.errcode = errno,final);

    /*解析参数。*/
    abcdk_getargs(ctx.args, argc, argv, "--");

    /*记录日志。*/
    abcdk_openlog(NULL, LOG_INFO, 1);

final:
    
    abcdk_tree_free(&ctx.args);

    return ctx.errcode;
}