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
#include "abcdkutil/mman.h"

#define DEFAULT_OS_RELEASE_FILE "/etc/os-release"

void _abcdklsb_print_usage(abcdk_tree_t *args, int only_version)
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

    fprintf(stderr, "\n%s [ --key < NAME > ] [ OPTIONS ] \n",name);

    fprintf(stderr, "\n%s \n",name);

    fprintf(stderr, "\nOPTIONS:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\tShow this help message and exit.\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\tOutput version information and exit.\n");

    fprintf(stderr, "\n\t--key < NAME >\n");
    fprintf(stderr, "\t\tThe name of the key. default: all\n");

    fprintf(stderr, "\n\t--short\n");
    fprintf(stderr, "\t\tShow requested information in short format.\n");

    fprintf(stderr, "\n\t--os-release-file < FILE >\n");
    fprintf(stderr, "\t\tCompatible os-release file. default: %s\n",DEFAULT_OS_RELEASE_FILE);
}

void _abcdklsb_work(abcdk_tree_t *args)
{
    const char *osinfo_file;
    const char *key;
    const char *value;
    int val_short;
    abcdk_tree_t *osinfo_args = NULL;
    abcdk_allocator_t *osinfo_mem = NULL;

    osinfo_file = abcdk_option_get(args, "--os-release-file", 0,DEFAULT_OS_RELEASE_FILE);
    key = abcdk_option_get(args, "--key", 0, "");
    val_short = (abcdk_option_exist(args, "--short") ? 1 : 0);

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
        //fprintf(stdout, "%s\n", osinfo_mem->pptrs[0]);
        abcdk_option_fprintf(stdout,osinfo_args,"=");
    }

final:

    abcdk_allocator_unref(&osinfo_mem);
    abcdk_tree_free(&osinfo_args);
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
        _abcdklsb_print_usage(args, 0);
    }
    else if (abcdk_option_exist(args, "--version"))
    {
        _abcdklsb_print_usage(args, 1);
    }
    else
    {
        _abcdklsb_work(args);
    }

final:

    abcdk_tree_free(&args);

    return errno;
}