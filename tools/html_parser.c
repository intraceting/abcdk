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


void _abcdkhtml_print_usage(abcdk_tree_t *args, int only_version)
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

    fprintf(stderr, "\n%s [ --html < FILE > ] [ OPTIONS ] \n",name);

    fprintf(stderr, "\n%s \n",name);

    fprintf(stderr, "\nOPTIONS:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\tShow this help message and exit.\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\tOutput version information and exit.\n");

    fprintf(stderr, "\n\t--html < FILE >\n");
    fprintf(stderr, "\t\tThe HTML file.\n");

    fprintf(stderr, "\n\t--robots < FILE >\n");
    fprintf(stderr, "\t\tThe robots file.\n");

    fprintf(stderr, "\n\t--user-agent < NAME >\n");
    fprintf(stderr, "\t\tUA name. default: *\n");

    fprintf(stderr, "\n\t--tag < NAME >\n");
    fprintf(stderr, "\t\tThe name of the tag.\n");

    fprintf(stderr, "\n\t--attr < KEY >\n");
    fprintf(stderr, "\t\tThe key of the attributes.\n");

    fprintf(stderr, "\n\t--cts\n");
    fprintf(stderr, "\t\tCompress space characters.\n");

}

int _abcdkhtml_match_tag(abcdk_tree_t *args, const char *name)
{
    const char *p = NULL;
    ssize_t c = abcdk_option_count(args, "--tag");

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

    for (ssize_t i = 0; i < c; i++)
    {
        p = abcdk_option_get(args, "--tag", i, "");
        if (abcdk_strcmp(p, key, 0) == 0)
            return 0;
    }

    return FNM_NOMATCH;
}

void _abcdkhtml_printf(abcdk_tree_t *html,abcdk_tree_t *rbts,abcdk_tree_t *args)
{
    abcdk_tree_t *tag_p = NULL;
    abcdk_tree_t *attr_p = NULL;
    int cts = 0;
    int chk;
    
    cts = abcdk_option_exist(args,"--cts");

    tag_p = abcdk_tree_child(html,1);
    while (tag_p)
    {
        chk = _abcdkhtml_match_tag(args, (char *)tag_p->alloc->pptrs[ABCDK_HTML_KEY]);
        if (chk == 0)
        {
            fprintf(stdout,"%s\n",tag_p->alloc->pptrs[ABCDK_HTML_VALUE]);
        }

        /*Next*/
        tag_p = abcdk_tree_sibling(tag_p, 0);
    }

    /*Clear errno.*/
    errno = 0;
}

void _abcdkhtml_work(abcdk_tree_t *args)
{
    int err = 0;
    abcdk_tree_t *html = NULL;
    abcdk_tree_t *rbts = NULL;
    const char *html_file = NULL;
    const char *rbts_file = NULL;
    const char *user_agent = NULL;
    

    html_file = abcdk_option_get(args, "--html", 0,NULL);
    rbts_file = abcdk_option_get(args, "--robots", 0,NULL);
    user_agent = abcdk_option_get(args, "--user-agent", 0,"*");


    if (!html_file || !*html_file)
    {
        syslog(LOG_ERR, "'--html FILE' cannot be omitted.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (access(html_file, R_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.",html_file,strerror(errno));
        goto final;
    }

    if (rbts_file)
    {
        if (access(rbts_file, R_OK) != 0)
        {
            syslog(LOG_WARNING, "'%s' %s.", rbts_file, strerror(errno));
            goto final;
        }
        else
        {
            rbts = abcdk_robots_parse_file(rbts_file,user_agent);
        }
    }

    html = abcdk_html_parse_file(html_file);
    if(!html)
    {
        syslog(LOG_WARNING, "'%s' Cannot parse, or is not in HTML format.", rbts_file);
        goto final;
    }

    _abcdkhtml_printf(html,rbts,args);

final:

    err = errno;
    abcdk_tree_free(&html);
    abcdk_tree_free(&rbts);
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