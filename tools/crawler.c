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
#include "abcdkutil/general.h"
#include "abcdkutil/getargs.h"
#include "abcdkutil/html.h"
#include "abcdkutil/robots.h"

void _abcdkcrawler_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    /*Clear errno.*/
    errno = 0;

    abcdk_proc_basename(name);

#ifdef BUILD_VERSION_DATETIME
    fprintf(stderr, "\n%s Build %s\n", name, BUILD_VERSION_DATETIME);
#endif //BUILD_VERSION_DATETIME

    fprintf(stderr, "\n%s Version %d.%d\n", name, ABCDK_VERSION_MAJOR, ABCDK_VERSION_MINOR);

    if (only_version)
        return;

    fprintf(stderr, "\nSYNOPSIS:\n");

    fprintf(stderr, "\n%s [ --html < FILE > ] [ OPTIONS ] \n", name);

    fprintf(stderr, "\n%s \n", name);

    fprintf(stderr, "\nOPTIONS:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\tShow this help message and exit.\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\tOutput version information and exit.\n");

    fprintf(stderr, "\n\t--html < FILE >\n");
    fprintf(stderr, "\t\tThe HTML file.\n");

    fprintf(stderr, "\n\t--tag < NAME >\n");
    fprintf(stderr, "\t\tThe name of the tag.\n");

    fprintf(stderr, "\n\t--attr < KEY >\n");
    fprintf(stderr, "\t\tThe key of the attributes.\n");

    fprintf(stderr, "\n\t--cts\n");
    fprintf(stderr, "\t\tCompress space characters.\n");
}

int _abcdkcrawler_match_tag(abcdk_tree_t *args, const char *name)
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

int _abcdkcrawler_match_attr(abcdk_tree_t *args, const char *key)
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

/**/
const char *_abcdkcrawler_cntrl_replace(char *text, char c)
{
    if(!text)
        return "";

    char *tmp = text;
    while (*tmp)
    {
        if (iscntrl(*tmp))
            *tmp = c;

        tmp += 1;
    }

    return text;
}

int _abcdkcrawler_printf_cb(size_t deep, abcdk_tree_t *node, void *opaque)
{
    abcdk_tree_t *args = NULL;
    int tag_short = 0;
    int attr_short = 0;
    int chk;

    args = (abcdk_tree_t*)opaque;
    tag_short = abcdk_option_exist(args,"--tag-short");
    attr_short = abcdk_option_exist(args,"--attr-short");
    

    if (deep == 0)
        return 1;

    if (deep == 1)
    {
        chk = _abcdkcrawler_match_tag(args, (char *)node->alloc->pptrs[ABCDK_HTML_KEY]);
        if (chk != 0)
            return 0;

        if (tag_short)
        {
            abcdk_tree_fprintf(stdout, deep, node, "%s\n",
                               _abcdkcrawler_cntrl_replace(ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_HTML_VALUE], 0), ' '));
        }
        else
        {
            abcdk_tree_fprintf(stdout, deep, node, "%s=%s\n",
                               ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_HTML_KEY], 0),
                               _abcdkcrawler_cntrl_replace(ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_HTML_VALUE], 0), ' '));
        }

        return 1;
    }

    if (deep == 2)
    {
        chk = _abcdkcrawler_match_attr(args, (char *)node->alloc->pptrs[ABCDK_HTML_KEY]);

        if (chk == 0)
        {
            if (attr_short)
            {
                abcdk_tree_fprintf(stdout, deep, node, "%s\n",
                                   ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_HTML_VALUE], 0));
            }
            else
            {
                abcdk_tree_fprintf(stdout, deep, node, "%s=%s\n",
                                   ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_HTML_KEY], 0),
                                   ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_HTML_VALUE], 0));
            }
        }

        return 1;
    }
}

void _abcdkcrawler_work(abcdk_tree_t *args)
{
    int err = 0;
    abcdk_tree_t *html = NULL;
    const char *file = NULL;

    file = abcdk_option_get(args, "--html", 0, NULL);

    if (!file || !*file)
    {
        syslog(LOG_ERR, "'--html FILE' cannot be omitted.");
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
        syslog(LOG_WARNING, "'%s' Cannot parse, or is not in HTML format.", file);
        goto final;
    }

    abcdk_tree_iterator_t it = {0, _abcdkcrawler_printf_cb, args};
    abcdk_tree_scan(html, &it);

final:

    err = errno;
    abcdk_tree_free(&html);
    errno = err;
}

int main(int argc, char **argv)
{
    abcdk_tree_t *args;

    args = abcdk_tree_alloc3(1);
    if (!args)
        goto final;

    abcdk_getargs(args, argc, argv, "--");
    //abcdk_option_fprintf(stderr, args);

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(args, "--help"))
    {
        _abcdkcrawler_print_usage(args, 0);
    }
    else if (abcdk_option_exist(args, "--version"))
    {
        _abcdkcrawler_print_usage(args, 1);
    }
    else
    {
        _abcdkcrawler_work(args);
    }

final:

    abcdk_tree_free(&args);

    return errno;
}