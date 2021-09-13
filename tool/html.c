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
#include "abcdk-util/html.h"

void _abcdkhtml_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);
    fprintf(stderr, "\n%s 版本 %d.%d-%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);

    if (only_version)
        ABCDK_ERRNO_AND_RETURN0(0);

    fprintf(stderr, "\n摘要:\n");

    fprintf(stderr, "\n%s [ --html < FILE > ] [ OPTIONS ] \n", name);

    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的HTML解析工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\t显示版本信息。\n");

    fprintf(stderr, "\n\t--html < FILE >\n");
    fprintf(stderr, "\t\tHTML文件(包括路径)。\n");

    fprintf(stderr, "\n\t--align-left\n");
    fprintf(stderr, "\t\t左对齐。默认：树型\n");

    fprintf(stderr, "\n\t--tag < NAME >\n");
    fprintf(stderr, "\t\tTAG名称，用于筛选器。默认: a img video\n");

    fprintf(stderr, "\n\t--tag-short\n");
    fprintf(stderr, "\t\t输出TAG短格式。\n");

    fprintf(stderr, "\n\t--tag-hide\n");
    fprintf(stderr, "\t\t隐藏TAG信息。\n");

    fprintf(stderr, "\n\t--attr < KEY >\n");
    fprintf(stderr, "\t\tATTR名称，用于筛选器。默认: src href\n");

    fprintf(stderr, "\n\t--attr-short\n");
    fprintf(stderr, "\t\t输出ATTR短格式。\n");

    fprintf(stderr, "\n\t--attr-hide\n");
    fprintf(stderr, "\t\t隐藏ATTR信息。\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到指定的文件(包括路径)。默认：终端\n");
    
    ABCDK_ERRNO_AND_RETURN0(0);
}

int _abcdkhtml_match_tag(abcdk_tree_t *args, const char *name)
{
    const char *p = NULL;
    ssize_t c = abcdk_option_count(args, "--tag");

    if (c <= 0)
        return 0;

    for (ssize_t i = 0; i < c; i++)
    {
        p = abcdk_option_get(args, "--tag", i, "");
        if (abcdk_strcmp(p, name, 0) == 0)
            return 0;
    }

    return FNM_NOMATCH;
}

int _abcdkhtml_match_attr(abcdk_tree_t *args, const char *key)
{
    const char *p = NULL;
    ssize_t c = abcdk_option_count(args, "--attr");

    if (c <= 0)
        return 0;

    for (ssize_t i = 0; i < c; i++)
    {
        p = abcdk_option_get(args, "--attr", i, "");
        if (abcdk_strcmp(p, key, 0) == 0)
            return 0;
    }

    return FNM_NOMATCH;
}

void _abcdkhtml_printf(size_t depth, const abcdk_tree_t *node, int only_val, int align_left,int hide)
{
    int have_val = (node->alloc->pptrs[ABCDK_HTML_VALUE]?1:0);

    if (hide)
    {
        if (!align_left)
            abcdk_tree_fprintf(stdout, depth, node, "\n");
        else 
            fprintf(stdout, "%s\n",(depth == 1 ? "\n" : ""));
    }
    else
    {
        if (!align_left)
        {

            abcdk_tree_fprintf(stdout, depth, node, "%s%s%s\n",
                               (only_val ? "" : (char *)node->alloc->pptrs[ABCDK_HTML_KEY]),
                               (only_val ? "" : "="),
                               (!have_val ? "" : (char *)node->alloc->pptrs[ABCDK_HTML_VALUE]));
        }
        else
        {
            fprintf(stdout, "%s%s%s%s\n",
                    (depth == 1 ? "\n" : ""),
                    (only_val ? "" : (char *)node->alloc->pptrs[ABCDK_HTML_KEY]),
                    (only_val ? "" : "="),
                    (!have_val ? "" : (char *)node->alloc->pptrs[ABCDK_HTML_VALUE]));
        }
    }
}

int _abcdkhtml_printf_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_tree_t *args = (abcdk_tree_t*)opaque;
    int tag_hide = 0;
    int tag_short = 0;
    int attr_hide = 0;
    int attr_short = 0;
    int align_left = 0;
    int chk;

    tag_hide = abcdk_option_exist(args,"--tag-hide");
    tag_short = abcdk_option_exist(args,"--tag-short");
    attr_hide = abcdk_option_exist(args,"--attr-hide");
    attr_short = abcdk_option_exist(args,"--attr-short");
    align_left = abcdk_option_exist(args,"--align-left");

    /*已经结束。*/
    if(depth == SIZE_MAX)
        return -1;
    
    if (depth == 0)
        return 1;

    if (depth == 1)
    {
        chk = _abcdkhtml_match_tag(args, (char *)node->alloc->pptrs[ABCDK_HTML_KEY]);
        if (chk != 0)
            ABCDK_ERRNO_AND_RETURN1(0,0);

        _abcdkhtml_printf(depth,node,tag_short,align_left,tag_hide);

        return 1;
    }

    if (depth == 2)
    {
        chk = _abcdkhtml_match_attr(args, (char *)node->alloc->pptrs[ABCDK_HTML_KEY]);

        if (chk == 0)
            _abcdkhtml_printf(depth,node,attr_short,align_left,attr_hide);

        return 1;
    }

    return 1;
}

void _abcdkhtml_work(abcdk_tree_t *args)
{
    int err = 0;
    abcdk_tree_t *html = NULL;
    const char *file = NULL;
    const char *outfile = NULL;

    file = abcdk_option_get(args, "--html", 0, NULL);
    outfile = abcdk_option_get(args, "--output", 0, NULL);

    if(!abcdk_option_exist(args, "--tag"))
    {
        abcdk_option_set(args,"--tag","a");
        abcdk_option_set(args,"--tag","img");
        abcdk_option_set(args,"--tag","video");
    }

    if(!abcdk_option_exist(args, "--attr"))
    {
        abcdk_option_set(args,"--attr","src");
        abcdk_option_set(args,"--attr","href");
    }

    /*Clear errno.*/
    errno = 0;

    if (!file || !*file)
    {
        syslog(LOG_ERR, "'--html FILE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (access(file, R_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.", file, strerror(errno));
        goto final;
    }

    html = abcdk_html_parse_file(file);
    if (!html)
    {
        syslog(LOG_WARNING, "'%s' 无法解析。", file);
        goto final;
    }

    if(!abcdk_tree_child(html,1))
    {
        syslog(LOG_WARNING, "'%s' 可能不是HTML格式。", file);
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

    abcdk_tree_iterator_t it = {0, _abcdkhtml_printf_cb, args};
    abcdk_tree_scan(html, &it);

final:

    err = errno;
    abcdk_tree_free(&html);
    errno = err;
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
        _abcdkhtml_print_usage(args, 0);
    }
    else if (abcdk_option_exist(args, "--version"))
    {
        _abcdkhtml_print_usage(args, 1);
    }
    else
    {
        _abcdkhtml_work(args);
    }

final:

    abcdk_tree_free(&args);

    return errno;
}