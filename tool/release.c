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
#include "abcdk/general.h"
#include "abcdk/getargs.h"
#include "abcdk/mman.h"

#define DEFAULT_OS_RELEASE_FILE "/etc/os-release"

void _abcdkrelease_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);
    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);

    if (only_version)
        ABCDK_ERRNO_AND_RETURN0(0);

    fprintf(stderr, "\n摘要:\n");

    fprintf(stderr, "\n%s [ --key < NAME > ] [ OPTIONS ] \n",name);

    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的系统信息查询工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\t显示版本信息。\n");

    fprintf(stderr, "\n\t--key < NAME >\n");
    fprintf(stderr, "\t\tKEY名称，用于筛选器。默认: 全部\n");

    fprintf(stderr, "\n\t--short\n");
    fprintf(stderr, "\t\t输出短格式。\n");

    fprintf(stderr, "\n\t--os-release < FILE >\n");
    fprintf(stderr, "\t\t指定兼容的os-release文件. 默认: %s\n",DEFAULT_OS_RELEASE_FILE);

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdkrelease_work(abcdk_tree_t *args)
{
    const char *osinfo_file;
    const char *key;
    const char *value;
    int val_short;
    abcdk_tree_t *osinfo_args = NULL;
    abcdk_allocator_t *osinfo_mem = NULL;

    osinfo_file = abcdk_option_get(args, "--os-release", 0,DEFAULT_OS_RELEASE_FILE);
    key = abcdk_option_get(args, "--key", 0, "");
    val_short = (abcdk_option_exist(args, "--short") ? 1 : 0);

    /*Clear errno.*/
    errno = 0;

    if (access(osinfo_file, R_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.",osinfo_file,strerror(errno));
        goto final;
    }

    osinfo_mem = abcdk_mmap2(osinfo_file, 0, 0);
    if(!osinfo_mem)
        goto final;

    osinfo_args = abcdk_tree_alloc3(1);
    if(!osinfo_args)
        goto final;

    abcdk_getargs_text(osinfo_args, (char*)osinfo_mem->pptrs[0],osinfo_mem->sizes[0] , '\n', 0, NULL, NULL);

    if (*key != '\0')
    {
        value = abcdk_option_get(osinfo_args, key, 0, "");
        if(val_short)
            fprintf(stdout, "%s\n", value);
        else
            fprintf(stdout, "%s=%s\n",key,value);

    }
    else
    {
        abcdk_option_fprintf(stdout,osinfo_args,"=");
    }

final:

    abcdk_allocator_unref(&osinfo_mem);
    abcdk_tree_free(&osinfo_args);
}

int main(int argc, char **argv)
{
    abcdk_tree_t *args;

    /*中文，UTF-8*/
    setlocale(LC_ALL,"zh_CN.UTF-8");

    args = abcdk_tree_alloc3(1);
    if (!args)
        goto final;

    abcdk_getargs(args, argc, argv, "--");
    //abcdk_option_fprintf(stderr, args);

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(args, "--help"))
    {
        _abcdkrelease_print_usage(args, 0);
    }
    else if (abcdk_option_exist(args, "--version"))
    {
        _abcdkrelease_print_usage(args, 1);
    }
    else
    {
        _abcdkrelease_work(args);
    }

final:

    abcdk_tree_free(&args);

    return errno;
}