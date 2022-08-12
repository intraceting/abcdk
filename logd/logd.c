/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "util/general.h"
#include "util/thread.h"
#include "util/getargs.h"
#include "util/openssl.h"
#include "shell/file.h"
#include "comm/easy.h"

typedef struct _abcdk_logd
{
    int errcode;
    abcdk_tree_t *args;

    const char *ca_file;
    const char *ca_path;
    int ca_check_crl;

} abcdk_logd_t;

void _abcdk_logd_print_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);
}

void _abcdk_logd_work(abcdk_logd_t *ctx)
{

}

int main(int argc, char **argv)
{
    abcdk_logd_t ctx = {0};

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

    if (abcdk_option_exist(ctx.args, "--help"))
        _abcdk_logd_print_usage(ctx.args);
    else
        _abcdk_logd_work(&ctx);

final:
    
    abcdk_tree_free(&args);

    return ctx.errcode;
}