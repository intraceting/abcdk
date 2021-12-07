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
#include "abcdk-util/general.h"
#include "abcdk-util/getargs.h"
#include "abcdk-util/robots.h"


void _abcdkrobots_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);
    fprintf(stderr, "\n%s 版本 %d.%d-%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);

    if (only_version)
        ABCDK_ERRNO_AND_RETURN0(0);

    fprintf(stderr, "\n摘要:\n");

    fprintf(stderr, "\n%s [ --robots < FILE > ] [--url < URL [URL ...] > ] [ OPTIONS ] \n", name);

    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t分析Robots文件，过滤被规则禁止的URL，仅输出规则允许的URL。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\t显示版本信息。\n");

    fprintf(stderr, "\n\t--robots < FILE >\n");
    fprintf(stderr, "\t\tRobots 文件名(包括路径)。\n");

    fprintf(stderr, "\n\t--user-agent < NAME >\n");
    fprintf(stderr, "\t\t代理名称。默认: *\n");

    fprintf(stderr, "\n\t--url < URL [URL ...] >\n");
    fprintf(stderr, "\t\t指定需要筛选的URL。\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到指定的文件(包括路径)。默认：终端\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

typedef struct _abcdkrobots_match_param
{
    int allow;
    int disallow;
    const char *url;
} abcdkrobots_match_param;

int _abcdkrobots_match_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdkrobots_match_param *param = (abcdkrobots_match_param *)opaque;
    const char *p = NULL;
    int chk;

    /*已经结束。*/
    if(depth == SIZE_MAX)
        return -1;

    if (depth == 0)
        ABCDK_ERRNO_AND_RETURN1(0, 1);

    p = abcdk_strstr(param->url, "://", 0);
    if (p)
    {
        p += 3;
        p = strchr(p, '/');
        if (!p)
            ABCDK_ERRNO_AND_RETURN1(0, -1);
    }
    else
    {
        p = param->url;
    }

    if (abcdk_strcmp(ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_KEY], 0), "Disallow", 0) == 0)
    {
        chk = abcdk_fnmatch(p, ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_VALUE], 0), 0, 0);
        if (chk == 0)
            param->disallow += 1;
    }
    else if (abcdk_strcmp(ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_KEY], 0), "Allow", 0) == 0)
    {
        chk = abcdk_fnmatch(p, ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_VALUE], 0), 0, 0);
        if (chk == 0)
            param->allow += 1;
    }

    ABCDK_ERRNO_AND_RETURN1(0, 1);
}

void _abcdkrobots_filter_url(abcdk_tree_t *rbts,abcdk_tree_t *args)
{
    abcdkrobots_match_param param = {0};
    ssize_t c = abcdk_option_count(args,"--url");

    for (ssize_t i = 0; i < c; i++)
    {
        param.disallow = 0;
        param.allow = 0;
        param.url = abcdk_option_get(args, "--url", i, "");

        abcdk_tree_iterator_t it = {0, _abcdkrobots_match_cb, &param};
        abcdk_tree_scan(rbts, &it);

        if (param.allow >= 1 || param.disallow == 0)
            fprintf(stdout, "%s\n", param.url);
    }

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdkrobots_work(abcdk_tree_t *args)
{
    int err = 0;
    abcdk_tree_t *rbts = NULL;
    const char *file = NULL;
    const char *agent = NULL;
    const char *outfile = NULL;

    file = abcdk_option_get(args, "--robots", 0, NULL);
    agent = abcdk_option_get(args, "--user-agent", 0, "*");
    outfile = abcdk_option_get(args, "--output", 0, NULL);

    /*Clear errno.*/
    errno = 0;


    if (!file || !*file)
    {
        syslog(LOG_ERR, "'--robots FILE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_RETURN0(EINVAL);
    }

    if (access(file, R_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s。", file, strerror(errno));
        return;
    }

    rbts = abcdk_robots_parse_file(file,agent);
    if (!rbts)
    {
        syslog(LOG_WARNING, "'%s' 解析失败。", file);
        return;
    }

    if(!abcdk_tree_child(rbts,1))
    {
        syslog(LOG_WARNING, "规则内未包含指定的代理名称'%s'。", agent);
        ABCDK_ERRNO_AND_GOTO1(EPERM, final);
    }

    if(outfile && *outfile)
    {
        if(abcdk_reopen(STDOUT_FILENO,outfile,1,0,1)<0)
        {
            syslog(LOG_WARNING, "'%s' %s.", outfile, strerror(errno));
            goto final;
        }
    }

    _abcdkrobots_filter_url(rbts,args);

    fflush(stdout);
    
final:

    abcdk_tree_free(&rbts);
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
        _abcdkrobots_print_usage(args, 0);
    }
    else if (abcdk_option_exist(args, "--version"))
    {
        _abcdkrobots_print_usage(args, 1);
    }
    else
    {
        _abcdkrobots_work(args);
    }

final:

    abcdk_tree_free(&args);

    return errno;
}