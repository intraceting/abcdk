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
#include "entry.h"

/** 工具集合。*/
static struct _abcdk_vmtx_entry
{
    /** 名字。*/
    const char *name;

    /** 
     * 回调函数。
     * 
     * @return 出错码
    */
    int (*entry_cb)(abcdk_tree_t *args);
}abcdk_vmtx_entry[] = {
    {"server",abcdk_vmtx_server},
    {"client",abcdk_vmtx_client}
};

void _abcdk_vmtx_print_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);

    fprintf(stderr, "\n命令：\n");
    fprintf(stderr, "\n\t");

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_vmtx_entry); i++)
    {
        fprintf(stderr, "%s ",abcdk_vmtx_entry[i].name);
    }

    fprintf(stderr, "\n");

    fprintf(stderr, "\n示例：\n");
    fprintf(stderr, "\n\t%s < CMD > --help\n", name);
    fprintf(stderr, "\n\t%s < CMD > [ ... ]\n", name);
}

struct _abcdk_vmtx_entry *_abcdk_vmtx_entry_find(abcdk_tree_t *args)
{
    const char *name_p = abcdk_option_get(args,"--",1,NULL);

    if(!name_p)
        return NULL;
    
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_vmtx_entry); i++)
    {
        if(abcdk_strcmp(abcdk_vmtx_entry[i].name,name_p,0)==0)
            return &abcdk_vmtx_entry[i];
    }

    return NULL;
}

int _abcdk_vmtx_dispatch(abcdk_tree_t *args)
{
    int errcode = 0;
    struct _abcdk_vmtx_entry *entry_p = NULL;

    entry_p = _abcdk_vmtx_entry_find(args);

    if (!entry_p)
    {
        _abcdk_vmtx_print_usage();
        ABCDK_ERRNO_AND_GOTO1(errcode = EINVAL, final);
    }

    errcode = entry_p->entry_cb(args);

final:

    return errcode;
}

int main(int argc, char **argv)
{
    abcdk_tree_t *args = NULL;
    int errcode = 0;

    /*中文；UTF-8。*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    /*随机数种子。*/
    srand(time(NULL));

    /*记录日志。*/
    abcdk_log_open(NULL, LOG_INFO, 1);

    /*申请参数存储空间。*/
    args = abcdk_tree_alloc3(1);
    if (!args)
        ABCDK_ERRNO_AND_GOTO1(errcode = errno,final);
    
    /*解析参数。*/
    abcdk_getargs(args, argc, argv, "--");

    errcode = _abcdk_vmtx_dispatch(args);

final:
    
    abcdk_tree_free(&args);

    return errcode;
}