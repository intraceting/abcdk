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


void _abcdk_vmc_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\n\t--master-address < ADDRESS [ ADDRESS ] >\n");
    fprintf(stderr, "\t\t主管理节点地址(IPv4,IPv6)。\n");
    fprintf(stderr, "\n\t\tIPv4：Address:Port\n");
    fprintf(stderr, "\t\tIPv6：[Address]:Port\n");
}

void _abcdk_vmc_dowork(abcdk_vmc_t *ctx)
{

}

int main(int argc, char **argv)
{
    abcdk_vmc_t ctx = {0};

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
    abcdk_log_open(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_vmc_usage();
    }
    else 
    {
        _abcdk_vmc_dowork(&ctx);
    }

final:
    
    abcdk_tree_free(&ctx.args);

    return ctx.errcode;
}