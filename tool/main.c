/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "entry.h"

/** 工具集合。*/
static struct _abcdk_tool_entry
{
    /** 名字。*/
    const char *name;

    /** 
     * 回调函数。
     * 
     * @return 出错码。
    */
    int (*func_cb)(abcdk_tree_t *args);
}abcdk_tool_entry[] = {
    {"serial",abcdk_tool_serial},
    {"robots",abcdk_tool_robots},
    {"odbc",abcdk_tool_odbc},
    {"mtx",abcdk_tool_mtx},
    {"mt",abcdk_tool_mt},
    {"mp4juicer",abcdk_tool_mp4juicer},
    {"mp4dump",abcdk_tool_mp4dump},
    {"html",abcdk_tool_html},
    {"hexdump",abcdk_tool_hexdump},
    {"json",abcdk_tool_json},
    {"lsscsi",abcdk_tool_lsscsi},
    {"archive",abcdk_tool_archive},
    {"lsmmc",abcdk_tool_lsmmc},
    {"basecode",abcdk_tool_basecode},
    {"logd",abcdk_tool_logd}
};

void _abcdk_tool_print_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);

    fprintf(stderr, "\n命令：\n");
    fprintf(stderr, "\n\t");

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_tool_entry); i++)
    {
        fprintf(stderr, "%s ",abcdk_tool_entry[i].name);
    }

    fprintf(stderr, "\n");

    fprintf(stderr, "\n示例：\n");
    fprintf(stderr, "\n\t%s < CMD > --help\n", name);
    fprintf(stderr, "\n\t%s < CMD > [ ... ]\n", name);
}

struct _abcdk_tool_entry *_abcdk_tool_entry_find(abcdk_tree_t *args)
{
    const char *name_p = abcdk_option_get(args,"--",1,NULL);

    if(!name_p)
        return NULL;
    
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_tool_entry); i++)
    {
        if(abcdk_strcmp(abcdk_tool_entry[i].name,name_p,0)==0)
            return &abcdk_tool_entry[i];
    }

    return NULL;
}

int _abcdk_tool_dispatch(abcdk_tree_t *args)
{
    int errcode = 0;
    struct _abcdk_tool_entry *entry_p = NULL;

    entry_p = _abcdk_tool_entry_find(args);

    if (!entry_p)
    {
        _abcdk_tool_print_usage();
        ABCDK_ERRNO_AND_GOTO1(errcode = EINVAL, final);
    }

    errcode = entry_p->func_cb(args);

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

    /*申请参数存储空间。*/
    args = abcdk_tree_alloc3(1);
    if (!args)
        ABCDK_ERRNO_AND_GOTO1(errcode = errno,final);
    
    /*解析参数。*/
    abcdk_getargs(args, argc, argv, "--");

    errcode = _abcdk_tool_dispatch(args);

final:
    
    abcdk_tree_free(&args);

    return errcode;
}