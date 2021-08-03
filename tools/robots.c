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
#include "abcdkutil/robots.h"


void _abcdkrobots_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

#ifdef BUILD_VERSION_DATETIME
    fprintf(stderr, "\n%s Build %s\n", name, BUILD_VERSION_DATETIME);
#endif //BUILD_VERSION_DATETIME

    fprintf(stderr, "\n%s Version %d.%d\n", name, ABCDK_VERSION_MAJOR, ABCDK_VERSION_MINOR);

    if (only_version)
        ABCDK_ERRNO_AND_RETURN0(0);

    fprintf(stderr, "\nSYNOPSIS:\n");

    fprintf(stderr, "\n%s [ --robots < FILE > ] [--url < NAME [ NAME ... ] > ] [ OPTIONS ] \n", name);

    fprintf(stderr, "\n%s \n", name);

    fprintf(stderr, "\nOPTIONS:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\tShow this help message and exit.\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\tOutput version information and exit.\n");

    fprintf(stderr, "\n\t--robots < FILE >\n");
    fprintf(stderr, "\t\tThe robots file.\n");

    fprintf(stderr, "\n\t--user-agent < NAME >\n");
    fprintf(stderr, "\t\tUA name. default: *\n");

    fprintf(stderr, "\n\t--url < NAME >\n");
    fprintf(stderr, "\t\tThe URL used for the filter. \n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\tOutput to the specified file.\n");

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
    if(depth == UINTMAX_MAX)
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
        syslog(LOG_ERR, "'--robots FILE' can not be omitted.");
        ABCDK_ERRNO_AND_RETURN0(EINVAL);
    }

    if (access(file, R_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.", file, strerror(errno));
        return;
    }

    rbts = abcdk_robots_parse_file(file,agent);
    if (!rbts)
    {
        syslog(LOG_WARNING, "'%s' can not parsed.", file);
        return;
    }

    if(!abcdk_tree_child(rbts,1))
    {
        syslog(LOG_WARNING, "The user-agent '%s' not existent.", agent);
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

final:

    abcdk_tree_free(&rbts);
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